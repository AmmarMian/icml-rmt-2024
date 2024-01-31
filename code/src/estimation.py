import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from joblib import Parallel, delayed

from .covariance import analytical_shrinkage_estimator


class Covariances_LWnonlinear(BaseEstimator, TransformerMixin):
    """
    Estimate several covariances with the analytical non linear shrinkage estimator of Ledoit and Wolf.
    This sticks to pyRiemann format.

    Parameters
    ----------
    n_job : int, optional
        Number of jobs, by default 1.
    """
    def __init__(self, n_jobs=1) -> None:
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """
        Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : Covariances instance
            The Covariances instance.
        """
        return self
    
    def transform(self, X):
        """
        Estimate covariance matrices with the analytical non linear shrinkage estimator of Ledoit and Wolf.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        covmats : ndarray, shape (n_matrices, n_channels, n_channels)
            Covariance matrices.
        """
        n_matrices, n_channels, n_times = X.shape 
        covmats = np.zeros((n_matrices,n_channels,n_channels))
        if self.n_jobs==1:
            for i in range(n_matrices):
                covmats[i] = analytical_shrinkage_estimator(X[i].T,shrink=0)
        else:
            cov_list = Parallel(n_jobs=self.n_jobs)(
                    delayed(analytical_shrinkage_estimator)(x.T,shrink=0)
                    for x in X
                )
            for i, cov in enumerate(cov_list):
                covmats[i] = cov
        return covmats
    

class ERPAugmentation(BaseEstimator, TransformerMixin):
    def __init__(self, classes=None, svd=None, **kwds):
        """Init."""
        self.classes = classes
        self.svd = svd
        self.kwds = kwds

    def fit(self, X, y):
        """Fit.

        Estimate the prototyped responses for each class.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.

        Returns
        -------
        self : ERPCovariances instance
            The ERPCovariances instance.
        """
        if self.svd is not None:
            if not isinstance(self.svd, int):
                raise TypeError('svd must be None or int')
        if self.classes is not None:
            classes = self.classes
        else:
            classes = np.unique(y)

        self.P_ = []
        for c in classes:
            # Prototyped response for each class
            P = np.mean(X[y == c], axis=0)

            # Apply svd if requested
            if self.svd is not None:
                U, _, _ = np.linalg.svd(P)
                P = U[:, 0:self.svd].T @ P

            self.P_.append(P)

        self.P_ = np.concatenate(self.P_, axis=0)
        return self
    
    def transform(self, X):
        """Data augmentation for ERP data.
        Data are concatenated with mean ERPs.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        X_augmented : ndarray, shape (n_matrices, n_components, n_times)
            Augmented trials for ERP, where the size of matrices
            `n_components` is equal to `(1 + n_classes) x n_channels` if `svd`
            is None, and to `n_channels + n_classes x min(svd, n_channels)`
            otherwise.
        """
        n_matrices, n_channels, n_times = X.shape
        n_channels_proto, n_times_p = self.P_.shape
        if n_times_p != n_times:
            raise ValueError(
                f"X and P do not have the same n_times: {n_times} and {n_times_p}")
        X_augmented = np.empty((n_matrices, n_channels + n_channels_proto, n_times), dtype=X.dtype)
        for i in range(n_matrices):
            X_augmented[i] = np.concatenate((self.P_, X[i]), axis=0)
        return X_augmented


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
    # taken from : https://github.com/AlejandroSantorum/scikit-rmt/blob/main/skrmt/covariance/estimator.py
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
