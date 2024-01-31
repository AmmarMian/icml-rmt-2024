import numpy as np
import numpy.linalg as la


# not used, but...
def RMT_squaredFisherDistance_SCMs(SCM1: np.ndarray, SCM2: np.ndarray, n_samples1: int, n_samples2: int) -> float:
    """
    Random matrix theory based estimator of the square of the Fisher distance between two sample covariance matrices (SCM).
    It is divided by the dimension of the matrices.
    The distance between SCMs is corrected to evaluate the distance between the true covariance matrices.

    Parameters
    ----------
    SCM1 : ndarray
        First SCM, SPD matrix.
    SCM2 : ndarray
        Second SCM, SPD matrix.
    n_samples1 : int
        Number of samples used to compute SCM1.
    n_samples2 : int
        Number of samples used to compute SCM2.

    Returns
    -------
    float
        Estimated square of the Fisher distance between the true covariances corresponding to SCM1 and SCM2.
    """
    # some parameters
    n_features = SCM1.shape[0]
    c1 = n_features / n_samples1
    c2 = n_features / n_samples2
    eye = np.eye(n_features)
    # eigenvalue decomposition and some operations on eigenvalues
    iL = la.inv(la.cholesky(SCM1))
    l = la.eigvalsh(iL @ SCM2 @ iL.T)
    l_sqrt = np.sqrt(l)
    l_1 = 1/l
    l_ln_c1 = np.log((1-c1)*l)
    # some other eigenvalue decompositions
    baseMat = np.einsum('i,j->ij',l_sqrt,l_sqrt)
    diagL = np.diag(l)
    e = la.eigvalsh(diagL - baseMat/(n_features-n_samples1))
    z = la.eigvalsh(diagL - baseMat/n_samples2)
    # vector r and matrices M and N
    mat = np.einsum('i,j->ij',l,l_1)
    mat_ln = np.log(mat)
    denom = np.einsum('i,j->ij',l,np.ones((n_features,)))
    denom = denom - denom.swapaxes(-2,-1) + diagL
    r = l_ln_c1 * l_1
    M = (mat - 1 - mat_ln + .5*eye) / (denom**2)
    N = (mat_ln + eye) / denom
    # result
    return l_ln_c1@l_ln_c1/n_features \
            + 2*((c1+c2)/c1/c2-1)*((e-z)@M@(e-l) + (e-l)@r) \
            - 2/n_features*np.sum((e-z)@N) \
            - (1/c2-1)*np.log((1-c1)*(1-c2))**2 - 2*(1/c2-1)*(e-z)@r



def RMT_squaredFisherDistance_deterministic_SCMs(M: np.ndarray, SCMs: np.ndarray, n_samples: int) -> float:
    """
    Random matrix theory based estimator of the mean of the square of the Fisher distances between one deterministic SPD matrix
    and a set of sample covariance matrices (SCMs). It is also divided by the dimension of the matrices.
    Distances are corrected to evaluate the distance of the deterministic SPD matrix and the true covariance matrices corresponding to the SCMs.

    Parameters
    ----------
    M : ndarray, shape (n_features,n_features)
        Deterministic SPD matrix.
    SCMs : ndarray, shape (...,n_features,n_features)
        Set of sample covariance matrices.
    n_samples : int
        Number of samples used to compute the SCMs.

    Returns
    -------
    list of float, shape (...)
        RMT corrected square of Fisher distances between M and the SCMs.
    """
    # some parameters
    n_features = SCMs.shape[-1]
    c = n_features / n_samples
    multi_eye = np.zeros(SCMs.shape) + np.eye(n_features)
    # eigenvalue decomposition and some operations on eigenvalues
    iL = la.inv(la.cholesky(M))
    l = la.eigvalsh(iL @ SCMs @ iL.T)
    l_sqrt = np.sqrt(l)
    l_1 = 1/l
    l_ln = np.log(l)
    # some other eigenvalue decomposition
    diagL = np.einsum('...i,...ij->...ij',l,multi_eye)
    z = la.eigvalsh(diagL - np.einsum('...i,...j->...ij',l_sqrt,l_sqrt)/n_samples)
    # vector q and matrix Q
    mat = np.einsum('...i,j->...ij',l,np.ones((n_features,)))
    mat = mat - mat.swapaxes(-2,-1)
    Q = (np.einsum('...i,...ij->...ij', l, np.log(np.einsum('...i,...j->...ij',l,l_1))) - mat + multi_eye/2) / (mat**2 + diagL)
    # result
    return np.einsum('...i,...i->...', l_ln, l_ln)/n_features + 2*np.mean(l_ln,axis=-1) \
            - 2/n_features*np.einsum('...i,...ij->...', l-z, Q) - (1/c-1)*np.log(1-c)**2 \
            - 2*(1/c-1)*np.einsum('...i,...i->...', l-z, l_1*l_ln)



def RMT_mean_squaredFisherDistance_deterministic_SCMs(M: np.ndarray, SCMs: np.ndarray, n_samples: int) -> float:
    """
    Random matrix theory based estimator of the mean of the square of the Fisher distances between one deterministic SPD matrix
    and a set of sample covariance matrices (SCMs). It is also divided by the dimension of the matrices.
    Distances are corrected to evaluate the distance of the deterministic SPD matrix and the true covariance matrices corresponding to the SCMs.

    Parameters
    ----------
    M : ndarray
        Deterministic SPD matrix of shape (n_features,n_features).
    SCMs : ndarray
        Set of sample covariance matrices of shape (n_matrices,n_features,n_features).
    n_samples : int
        Number of samples used to compute the SCMs.

    Returns
    -------
    float
        Mean of the RMT corrected square of Fisher distances between M and the SCMs.
    """
    return np.mean(RMT_squaredFisherDistance_deterministic_SCMs(M,SCMs,n_samples))



def RMT_mean_squaredFisherDistance_deterministic_SCMs_rgrad(M: np.ndarray, SCMs: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Riemannian gradient of the random matrix theory based estimator of the mean of the square of the Fisher distances between
    one deterministic SPD matrix and a set of sample covariance matrices (SCMs). It is also divided by the dimension of the matrices.
    Distances are corrected to evaluate the distance of the deterministic SPD matrix and the true covariance matrices corresponding to the SCMs.

    Parameters
    ----------
    M : ndarray, shape (n_features,n_features)
        Deterministic SPD matrix.
    SCMs : ndarray, shape (n_matrices,n_features,n_features)
        Set of sample covariance matrices.
    n_samples : int
        Number of samples used to compute the SCMs.

    Returns
    -------
    ndarray, shape (n_features,n_features)
        Riemannian gradient of the mean of the RMT corrected square of Fisher distances between M and the SCMs.
    """
    # some parameters
    n_features = SCMs.shape[-1]
    c = n_features / n_samples
    multi_eye = np.zeros(SCMs.shape) + np.eye(n_features)
    one_vec = np.ones((n_features,))
    # eigenvalue decomposition and some operations on eigenvalues
    L = la.cholesky(M)
    iL = la.inv(L)
    l, U = la.eigh(iL @ SCMs @ iL.T)
    l_sqrt = np.sqrt(l)
    l_1 = 1/l
    l_ln = np.log(l)
    # some other eigenvalue decomposition
    diagL = np.einsum('...i,...ij->...ij',l,multi_eye)
    z, V = la.eigh(diagL - np.einsum('...i,...j->...ij',l_sqrt,l_sqrt)/n_samples)
    # vector q and matrix Q
    mat = np.einsum('...i,j->...ij',l,np.ones((n_features,)))
    mat = mat - mat.swapaxes(-2,-1)
    Q = (np.einsum('...i,...ij->...ij', l, np.log(np.einsum('...i,...j->...ij',l,l_1))) - mat + multi_eye/2) / (mat**2 + diagL)
    q = l_1*l_ln
    # gradient stuff
    delta = Q@one_vec/n_features + (1-c)/c*q
    delta_ = np.einsum('...ik,...ki->...i',
                        multi_eye-np.einsum('...i,...j->...ij',1/l_sqrt,l_sqrt)/n_samples,
                        np.einsum('...ik,...k,...jk->...ij',V,delta,V))
    A = -(np.einsum('...i,j->...ij',l_ln,one_vec)-np.einsum('i,...j->...ij',one_vec,l_ln)) \
        *(np.einsum('...i,j->...ij',l,one_vec)+np.einsum('i,...j->...ij',one_vec,l)) \
        /(mat**3+ multi_eye) \
        - 2/(2*diagL**2-mat**2)
    B = ((multi_eye - np.ones((n_features,n_features)))*np.einsum('i,...j->...ij',one_vec,l_1)) / (mat+multi_eye) \
        + 2 * (np.einsum('...i,j->...ij',l*l_ln,one_vec) - np.einsum('...i,...j->...ij',l,l_ln)) / (mat**3+multi_eye) \
        - 2 * np.ones((n_features,n_features)) / (mat**2-4*diagL**2)
    diagQ = (np.einsum('...ik,...i->...i',A,l-z) + np.einsum('...k,...ki->...i',l-z,B)) / n_features
    grad_l = (l_ln+one_vec)*l_1 / n_features - delta + delta_ - diagQ - (1-c)/c*(1-l_ln)*(l-z)*l_1**2
    # grad_l = (l_ln+one_vec)*l_1 / n_features
    return -2* L @ np.mean(np.einsum('...ik,...k,...jk->...ij',U,l*grad_l,U),axis=0) @ L.T



def squaredFisherDistance(M: np.ndarray, Ns: np.ndarray) -> float:
    """
    Square of the Gaussian Fisher distance between two SPD matrices.
    It is divided by the dimension of the matrices.

    Parameters
    ----------
    M : jnp.ndarray, shape (n_features,n_features)
        SPD matrix of.
    Ns : jnp.ndarray, shape (...,n_features,n_features)
        SPD matrix or set of SPD matrices.

    Returns
    -------
    list of floats, shape (...)
        list of squared distances between M and matrices in Ns.
    """
    n_features = M.shape[0]
    iL = la.inv(la.cholesky(M))
    l = la.eigvalsh(iL @ Ns @ iL.T)
    l_ln = np.log(l)
    return np.einsum('...i,...i->...', l_ln, l_ln)/n_features

def mean_squaredFisherDistance(M: np.ndarray, Ns: np.ndarray) -> float:
    """
    Mean of the square of the Gaussian Fisher distances between a SPD matrix and a set of SPD matrices.
    It is divided by the dimension of the matrices.

    Parameters
    ----------
    M : jnp.ndarray
        SPD matrix of shape (n_features,n_features).
    Ns : jnp.ndarray
        Set of SPD matrices of shape (n_matrices,n_features,n_features).

    Returns
    -------
    float
        Mean of the square of the distances between M and all Ns.
    """
    return np.mean(squaredFisherDistance(M,Ns))

def mean_squaredFisherDistance_egrad(M: np.ndarray, Ns:np.ndarray) -> np.ndarray:
    """
    Euclidean gradient of the mean of the square of the Gaussian Fisher distances between a SPD matrix and a set of SPD matrices.
    It is divided by the dimension of the matrices.

    Parameters
    ----------
    M : jnp.ndarray
        SPD matrix of shape (n_features,n_features).
    Ns : jnp.ndarray
        Set of SPD matrices of shape (n_matrices,n_features,n_features).

    Returns
    -------
    jnp.ndarray
        Euclidean gradient at M of the mean of the square of the distances between M and all Ns.
    """
    n_matrices, n_features, _ = Ns.shape
    L = la.cholesky(M)
    iL = la.inv(L)
    l, U = la.eigh(iL @ Ns @ iL.T)
    l_ln = np.log(l)
    return - 2*iL.T @ np.einsum('kil,kl,klj->ij',U,l_ln,U.swapaxes(-2,-1)) @ iL /n_features/n_matrices
