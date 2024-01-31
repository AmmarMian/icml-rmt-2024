import numpy as np



def SCM_estimator(data):
    """
    Sample covariance matrix (SCM) estimator.

    Parameters
    ----------
    data : ndarray
        data of shape (...,n_samples,n_features).

    Returns
    -------
    ndarray
        SCMs of data, symmetric positive definite matrices of shape (...,n_features,n_features).
    """
    return data.swapaxes(-2,-1)@data/data.shape[-2]


def analytical_shrinkage_estimator(data_mtx: np.ndarray, shrink: int = None) -> np.ndarray:
    """Estimates analytical shrinkage estimator.
    This estimator combines the best qualities of three different estimators:
    the speed of linear shrinkage, the accuracy of the well-known QuEST function
    and the transparency of the routine NERCOME. This estimator achieves this
    goal through nonparametric kernel estimation of the limiting spectral
    density of the sample eigenvalues and its Hilbert transform.

    Args:
        data_mtx (np.ndarray): data matrix containing n observations of size p, i.e.,
            data_mtx is a n times p matrix.
        shrink (int): number of degrees of freedom to substract.

    Returns:
        numpy array representing sample covariance matrix.
    References:
        Ledoit, O. and Wolf, M.
            "Analytical nonlinear shrinkage of large-dimensional covariance matrices".
            Annals of Statistics. 48.5 (2020): 3043-3065
    """
    # taken from : https://github.com/AlejandroSantorum/scikit-rmt/blob/main/skrmt/covariance/estimator.py (19/12/2023)
    n_size, p_size = data_mtx.shape

    if shrink is None:
        # demean data matrix
        data_mtx = data_mtx - data_mtx.mean(axis=0)
        # subtract one degree of freedom
        shrink=1
    # effective sample size
    n_size=n_size-shrink

    # get sample eigenvalues and eigenvectors, and sort them in ascending order
    sample = np.matmul(data_mtx.T, data_mtx)/n_size
    eigvals, eigvects = np.linalg.eig(sample)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvects = eigvects[:,order]

    # compute analytical nonlinear shrinkage kernel formula
    #eigvals = eigvals[max(0,p-n):p]
    repmat_eigs = np.tile(eigvals, (min(p_size,n_size), 1)).T
    h_list = n_size**(-1/3) * repmat_eigs.T

    eigs_div = np.divide((repmat_eigs-repmat_eigs.T), h_list)

    f_tilde=(3/4/np.sqrt(5))*np.mean(np.divide(np.maximum(1-eigs_div**2/5, 0), h_list), axis=1)

    hilbert_temp = (-3/10/np.pi)*eigs_div + \
                    (3/4/np.sqrt(5)/np.pi)*(1-eigs_div**2/5)*\
                        np.log(abs((np.sqrt(5)-eigs_div)/(np.sqrt(5)+eigs_div)))
    hilbert_temp[abs(eigs_div)==np.sqrt(5)] = (-3/10/np.pi) * eigs_div[abs(eigs_div)==np.sqrt(5)]
    hilbert = np.mean(np.divide(hilbert_temp, h_list), axis=1)

    # if p <= n: (we could improve it to support p>n case)
    d_tilde = np.divide(eigvals,
                        (np.pi*(p_size/n_size)*eigvals*f_tilde)**2 + \
                            (1-(p_size/n_size)-np.pi*(p_size/n_size)*eigvals*hilbert)**2
                       )

    # compute analytical nonlinear shrinkage estimator (sigma_tilde)
    return np.matmul(np.matmul(eigvects, np.diag(d_tilde)), eigvects.T)