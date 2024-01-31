# Utility functions  for generating data and other stuff

import os
import numpy as np
import numpy.linalg as la
from scipy.stats import norm, multivariate_normal
from sklearn.covariance import LedoitWolf, OAS
from .estimation import analytical_shrinkage_estimator
from .mean import geometric_mean
from .covariance import SCM_estimator
from .spd_manifold import SPD

import yaml
import logging
import datetime

from argparse import Namespace

def parse_args(args: Namespace, experiment: str):
    """Parse args from yaml or argparse. Priority is given to argparse."""

    if args.results_path is None:
        now = datetime.datetime.now()
        path = os.path.join(
                os.getcwd(), experiment,
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        args.results_path = path

    # Parse arguments for the simulation
    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = vars(args)

    for key, value in vars(args).items():
        if value is not None:
            logging.warning(f"Overwriting {key} with {value}")
            config[key] = value

    return config


def riemannian_dist2(mean, estimate):
    spd = SPD(mean.shape[0])
    return spd.dist(mean,estimate)**2


def multi_sym(matrices: np.ndarray) -> np.ndarray:
    """
    Returns the symmetrical parts of a set of matrices.
    Matrices are assumed to be on the two last axes.

    Parameters
    ----------
    matrices : ndarray
        Set of matrices of shape (..., n_features, n_features).

    Returns
    -------
    ndarray
        Set of symmetrical matrices of shape (..., n_features, n_features).
    """
    return 0.5*(matrices+matrices.swapaxes(-2,-1))


def random_SPD(n_features:int, condition_number:int=100) -> np.ndarray:
    """
    Random SPD matrix.

    Parameters
    ----------
    n_features : int
        Matrix size.
    condition_number : int, optional
        Condition number of generated matrix, by default 100.

    Returns
    -------
    np.ndarray
        SPD matrix of shape (n_features, n_features).
    """
    man = SPD(n_features)
    return man.random_point(condition_number=condition_number)


def _random_tangent_vectors(n_features:int, n_matrices:int, scale:float=0.1) -> np.ndarray:
    """
    Random tangent vectors in "canonical" format.
    Needs to do congruence with square root of point they are at to get "ambient" version.

    Parameters
    ----------
    n_features : int
        Size of the matrices.
    n_matrices : int
        Number of random matrices generated.
    scale : float, optional
        Variance of generated points around the zero matrix.
        Be careful, can very easily be too big to ensure numerical stability.
        By default 0.1.

    Returns
    -------
    np.ndarray
        Set of random symmetrical matrices centered around the zero matrix.
        Shape (n_matrices, n_features, n_features).
    """
    tangent_vectors = np.zeros((n_matrices,n_features,n_features))
    # generate indices of interest
    m_base_ix = np.arange(n_matrices)
    triu_indices = np.triu_indices(n_features,k=1)
    m_t_ix = np.repeat(m_base_ix,n_features*(n_features-1)//2)
    t_indices = (m_t_ix,) + (np.tile(triu_indices[0],n_matrices),) + (np.tile(triu_indices[1],n_matrices),)
    diag_indices = np.diag_indices(n_features)
    m_d_ix = np.repeat(m_base_ix,n_features)
    d_indices = (m_d_ix,) + (np.tile(diag_indices[0],n_matrices),) + (np.tile(diag_indices[1],n_matrices),)
    # fill tangent vectors with appropriate random values
    tangent_vectors[t_indices] = norm.rvs(size=n_matrices*(n_features*(n_features-1)//2),scale=scale)
    tangent_vectors = 2*multi_sym(tangent_vectors)
    tangent_vectors[d_indices] = norm.rvs(size=n_matrices*n_features,scale=scale)
    return tangent_vectors


def random_points_from_mean(geometric_mean:np.ndarray, n_matrices:int, scale:float=0.1, mode:str="exact") -> np.ndarray:
    """
    Returns a random set of SPD matrices from a given geometric mean. 

    Parameters
    ----------
    geometric_mean : ndarray
        Geometric mean - SPD matrix of shape (n_features, n_features).
    n_matrices : int
        number of matrices to be generated.
    scale : float, optional
        Variance of generated points around the mean.
        Be careful, can very easily be too big to ensure numerical stability.
        By default 0.1.
    mode : str, optional
        Either "exact" or "asymptotic".
        If "exact", the input geomeric mean is excatly the true geometric mean of the generated points.
        If "asymptotic", the input geomeric mean becomes the true geometric mean of the generated points as n_matrices tends to infinity.
        By default "exact".

    Returns
    -------
    ndarray
        Set of SPD matrices of shape (n_matrices, n_features, n_features).
    """
    n_features = geometric_mean.shape[0]
    # generate some tangent vectors in "canonical" format
    tangent_vectors = _random_tangent_vectors(n_features, n_matrices, scale)
    if mode=="exact":
        tangent_vectors = tangent_vectors - np.mean(tangent_vectors,axis=0)
    elif mode=="asymptotic":
        pass
    else:
        raise ValueError("Unknown mode for random_points_from_mean")
    # project tangent vectors onto the manifold
    ## diagonal indices for set of matrices of shape (n_matrices,n_features,n_features)
    m_base_ix = np.arange(n_matrices)
    m_d_ix = np.repeat(m_base_ix,n_features)
    diag_indices = np.diag_indices(n_features)
    d_indices = (m_d_ix,) + (np.tile(diag_indices[0],n_matrices),) + (np.tile(diag_indices[1],n_matrices),) 
    ## expm(tangent_vectors)
    d, U = la.eigh(tangent_vectors)
    ed = np.exp(d)
    D = np.zeros((n_matrices,n_features,n_features))
    D[d_indices] = ed.flatten()
    expm_tangent_vectors = U @ D @ U.swapaxes(-2,-1)
    ## actual projection
    L = la.cholesky(geometric_mean)
    return L @ expm_tangent_vectors @ L.T


def random_gaussian_data(covariances:np.ndarray, n_samples:int,
                         return_iterates: bool =  False) -> np.ndarray:
    """
    Generates random gaussian data from a set of covariance matrices.

    Parameters
    ----------
    covariances : ndarray
        SPD matrices of shape (n_matrices,n_features,n_features).
    n_samples : int
        Number of samples to generate for every covariance.
    return_iterates : bool, optional
        Whether to return all iterates of the algorithm, by default False.

    Returns
    -------
    ndarray
        Set of samples of shape (n_matrices,n_samples,n_features).

    list, optional
        Iterates of the algorithm
    """
    n_matrices, n_features, _ = covariances.shape
    data = np.zeros((n_matrices,n_samples,n_features))
    for mix in range(n_matrices):
        data[mix] = multivariate_normal.rvs(mean=np.zeros((n_features,)),cov=covariances[mix],size=n_samples)
    return data

def geometric_mean_2steps(data, init=None, cov_estimator="scm", return_iterates=False):
    n_matrices,n_samples,n_features = data.shape
    # generate covariances from data
    if cov_estimator=="scm":
        covariances = SCM_estimator(data)
    elif cov_estimator=="lw_linear":
        covariances = np.zeros((n_matrices,n_features,n_features))
        for mix in range(n_matrices):
            covariances[mix] = LedoitWolf(assume_centered=True).fit(data[mix]).covariance_
    elif cov_estimator=="oas":
        covariances = np.zeros((n_matrices,n_features,n_features))
        for mix in range(n_matrices):
            covariances[mix] = OAS(assume_centered=True).fit(data[mix]).covariance_
    elif cov_estimator=="lw_nonlinear":
        covariances = np.zeros((n_matrices,n_features,n_features))
        for mix in range(n_matrices):
            covariances[mix] = analytical_shrinkage_estimator(data[mix],shrink=0)
    else:
        raise ValueError("Unrecognized covariance estimator")
    # compute geometric mean
    return geometric_mean(covariances, init, return_iterates=return_iterates)
