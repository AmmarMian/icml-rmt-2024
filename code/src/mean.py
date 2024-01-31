import numpy as np
import numpy.linalg as la

from functools import partial

from scipy.linalg.lapack import dtrtri, dpptrf

from .covariance import SCM_estimator


def symm(x):
    return (x+x.swapaxes(-2,-1))/2


def RMT_geometric_mean(data:np.ndarray, n_dof=None,
                       init:np.ndarray=None, max_iterations:int=100,
                       tol:float=1e-6, return_iterates:bool=False):
    """
    Estimates the geometric mean of the true covariance matrices of some data by leveraging random matrix theory.

    Parameters
    ----------
    data : ndarray
        Dataset of shape (n_matrices,n_samples,n_features).
    init : ndarray, optional
        Initial guess, by default None.
        If None, initialization set to identity matrix.
    max_iterations : int, optional
        maximum number of iterations with pymanopt SteepestDescent.
        By default, 10.
    tol : float, optional
        tolerance for stopping criterion, by default 1e-6.
    return_iterates : bool, optional
        Whether to return all iterates, by default False.

    Returns
    -------
    ndarray
        Estimated geometric mean, shape (n_features,n_features).
    ndarray, optional
        Iterates of the algorithm, shape (n_iterations,n_features,n_features)
    """ 
    n_matrices, n_samples, n_features = data.shape
    if n_dof is None:
        n_dof = n_samples
    # compute SCMs
    SCMs = SCM_estimator(data)
    # some parameters
    c = n_features / n_dof
    trilix = np.tril_indices(n_features)
    # some initialization stuff
    if init is None:
        init = np.eye(n_features)
    M = init
    iterates = []
    iterates.append(M)
    err = 10
    it = 0
    # some other parameters
    oldcost = None
    one_vec = np.ones((n_features,))
    while (err > tol) and (it < max_iterations):
        # Cholesky and inverse of M
        L = np.zeros((n_features,n_features))
        L[trilix] = dpptrf(n_features,M[trilix])[0] 
        iL, _ = dtrtri(L, lower=1)
        # transformed SCMs
        trans_SCMs = iL @ SCMs @ iL.T
        # cost and grad at current iterate
        cost, grad_canonical = _aux_RMT_mean_cost_grad(trans_SCMs, n_features, n_dof, c, return_grad=True)
        # descent direction
        descent_dir = - symm(np.real(grad_canonical))
        d, W = la.eigh(descent_dir)
        trans_mats = W.T @ trans_SCMs @ W
        # stepsize
        alpha = _aux_linesearch(d,trans_mats, partial(_aux_RMT_mean_cost_grad, n_features=n_features, n_samples=n_dof, c=c, return_grad=False),cost,oldcost)
        alphad = alpha*d
        # new iterate
        tmp = L @ W
        newM = np.einsum('ik,k,kj->ij',tmp,one_vec + alphad + alphad**2/2,tmp.T)
        newM = symm(np.real(newM))
        # update some variables
        err = la.norm(M-newM)/la.norm(M)
        M = newM
        iterates.append(M)
        oldcost = cost
        it+=1
        # print(f'{it} \t {err}')
    if return_iterates:
        return M, iterates
    else:
        return M


def geometric_mean(matrices:np.ndarray, init:np.ndarray=None, max_iterations:int=100, return_iterates:bool=False):
    """
    Computes the geometric mean of a set of SPD matrices.

    Parameters
    ----------
    matrices : ndarray
        Set of SPD matrices of shape (n_matrices,n_features,n_features).
    init : ndarray, optional
        Initial guess, by default None.
        If None, initialization set to np.eye(n_features).
    max_iterations : int, optional
        maximum number of iterations with pymanopt SteepestDescent.
        By default, 100.
    return_iterates : bool, optional
        Whether to return all iterates, by default False.

    Returns
    -------
    ndarray
        Estimated geometric mean of shape (n_features,n_features).
    ndarray, optional
        Iterates of the algorithm, shape (n_iterations,n_features,n_features)
    """
    n_matrices, n_features, _ = matrices.shape
    trilix = np.tril_indices(n_features)
    # some initialization stuff
    if init is None:
        init = np.eye(n_features)
    M = init
    iterates = []
    iterates.append(M)
    err = 10
    tol = 1e-3
    it = 0
    # some other parameters
    oldcost = None
    one_vec = np.ones((n_features,))
    while (err > tol) and (it < max_iterations):
        # Cholesky and inverse of M
        L = np.zeros((n_features,n_features))
        L[trilix] = dpptrf(n_features,M[trilix])[0] 
        iL, _ = dtrtri(L, lower=1)
        # transformed matrices
        trans_matrices = iL @ matrices @ iL.T
        # cost and grad at current iterate
        cost, grad_canonical = _aux_mean_cost_grad(trans_matrices, n_features, return_grad=True)
        # descent direction
        descent_dir = - symm(np.real(grad_canonical))
        d, W = la.eigh(descent_dir)
        trans_mats = W.T @ trans_matrices @ W
        # stepsize
        alpha = _aux_linesearch(d,trans_mats, partial(_aux_mean_cost_grad, n_features=n_features, return_grad=False),cost,oldcost)
        alphad = alpha*d
        # new iterate
        tmp = L @ W
        newM = np.einsum('ik,k,kj->ij',tmp,one_vec + alphad + alphad**2/2,tmp.T)
        newM = symm(np.real(newM))
        # update some variables
        err = la.norm(M-newM)/la.norm(M)
        M = newM
        iterates.append(M)
        oldcost = cost
        it+=1
        # print(f'{it} \t {err}')
    if return_iterates:
        return M, iterates
    else:
        return M


def _aux_RMT_mean_cost_grad(mats, n_features, n_samples, c, return_grad=True):
    multi_eye = np.zeros(mats.shape) + np.eye(n_features)
    one_vec = np.ones((n_features,))
    # eigenvalue decomposition and some transformation of eigenvalues
    l, U = la.eigh(mats)
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
    # cost
    costs = np.einsum('...i,...i->...', l_ln, l_ln)/(2*n_features) + np.mean(l_ln,axis=-1) \
        - 1/n_features*np.einsum('...i,...ij->...', l-z, Q) - (1/c-1)*(np.log(1-c)**2)/2 \
        - (1/c-1)*np.einsum('...i,...i->...', l-z, q)
    cost = np.mean(costs)
    if return_grad:
        # gradient, canonical form (need congruence with L to get actual Riemannian gradient)
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
        grad_canonical = -np.mean(np.einsum('...ik,...k,...jk->...ij',U,l*grad_l,U), axis=0)
        return cost, grad_canonical
    else:
        return cost
    

def _aux_mean_cost_grad(mats, n_features, return_grad=True):
    # eigenvalue decomposition and some transformation of eigenvalues
    l, U = la.eigh(mats)
    l_ln = np.log(l)
    # cost
    costs = np.einsum('...i,...i->...', l_ln, l_ln)/(2*n_features)
    cost = np.mean(costs)
    if return_grad:
        # gradient, canonical form (need congruence with L to get actual Riemannian gradient)
        grad_l = l_ln/ n_features
        grad_canonical = -np.mean(np.einsum('...ik,...k,...jk->...ij',U,grad_l,U), axis=0)
        return cost, grad_canonical
    else:
        return cost

def _aux_linesearch(d, trans_mats, _aux_cost, cost, oldcost=None):
    alpha0=1
    optimism = 2
    sufficient_decrease = 1e-4
    contraction_factor = 0.5
    ls_maxit = 25
    # gradient norm
    norm_grad_2 = np.sum(d**2)
    # initialize stepsize
    if oldcost is None:
            alpha =  alpha0 / norm_grad_2**(0.5)
    else:
        alpha = - 2*(cost-oldcost) / norm_grad_2
        alpha *= optimism
    # initialize backtracking
    diag = (1 + alpha*d + alpha**2*d**2/2)**(-0.5)
    new_mats = np.einsum('i,...ij,j->...ij',diag,trans_mats,diag)
    newcost = _aux_cost(new_mats)
    ls_it = 0
    # perform backtracking
    while (newcost > cost - sufficient_decrease * alpha * norm_grad_2) and (ls_it < ls_maxit):
        alpha *= contraction_factor
        diag = (1 + alpha*d + alpha**2*d**2/2)**(-0.5)
        new_mats = np.einsum('i,...ij,j->...ij',diag,trans_mats,diag)
        newcost = _aux_cost(new_mats)
        ls_it += 1
    # if cost increases, reject step
    if newcost>cost:
        alpha = 0
    return alpha
