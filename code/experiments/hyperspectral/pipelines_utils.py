import numpy as np
import numpy.linalg as la
from numpy.lib.stride_tricks import sliding_window_view
from numpy import ndarray
from math import ceil

from scipy.optimize import linear_sum_assignment

from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin

from pyriemann.clustering import Kmeans
from pyriemann.utils.distance import distance

from joblib import Parallel, delayed

from typing import Callable, List


# Classes
# -------
class RemoveMeanImage(BaseEstimator, TransformerMixin):
    """A class to remove the mean of an image of shape (n_rows, n_columns, n_channels).
    """
    def __init__(self):
        pass
    def fit(self, X: ndarray, y=None):
        return self
    def transform(self, X: ndarray):
        return X - np.mean(X, axis=(0, 1))
    def fit_transform(self, X: ndarray, y=None):
        return self.fit(X).transform(X)


class PCAImage(BaseEstimator, TransformerMixin):
    """A class to apply PCA on an image of shape (n_rows, n_columns, n_channels).
    """

    def __init__(self, n_components: int):
        assert n_components > 0, 'Number of components must be positive.'
        self.n_components = n_components

    def fit(self, X: ndarray, y=None):
        return self

    def transform(self, X: ndarray):
        return pca_image(X, self.n_components)

    def fit_transform(self, X: ndarray, y=None):
        return self.fit(X).transform(X)


class SlidingWindowVectorize(BaseEstimator, TransformerMixin):
    """A class to apply a sliding windows and reshape the data to a shape of
    (n_samples, n_features).
    """

    def __init__(self, window_size: int, overlap: int = 0):
        assert window_size % 2 == 1, 'Window size must be odd.'
        assert overlap >= 0, 'Overlap must be positive.'
        assert overlap <= window_size//2, 'Overlap must be smaller or equal than int(window_size/2).'
        self.window_size = window_size
        self.overlap = overlap

    def fit(self, X: ndarray, y=None):
        return self

    def transform(self, X: ndarray):
        X = sliding_window_view(
                X,
                window_shape=(self.window_size, self.window_size),
                axis=(0, 1))
        if self.overlap is not None:
            if self.overlap > 0:
                X = X[::self.overlap, ::self.overlap]
        else:
            X = X[::self.window_size//2, ::self.window_size//2]
            self.overlap = self.window_size//2

        # Reshape to (n_pixels, n_samples, n_features) with n_pixels=axis0*axis1
        # n_samples=axis3*axis_4 and n_features=axis2
        X = X.reshape((-1, X.shape[2], X.shape[3]*X.shape[4]))
        return X
        
    def fit_transform(self, X: ndarray, y=None):
        return self.fit(X).transform(X)


class KmeansplusplusTransform(ClusterMixin, TransformerMixin):
    """A class to perform Kmeans++ clustering thanks to pyriemann Kmeans
    implementation. The transform allows to obtain the labels of the data.
    """

    def __init__(self, n_clusters: int, random_state: int = 42,
                max_iter: int = 100, tol: float = 1e-6,
                metric: str = 'riemann', n_jobs: int = 1, verbose: int = 0,
                 use_plusplus: bool = True, n_init: int = 1):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric
        self.distance = lambda x, y: distance(x, y, metric=self.metric)
        self.use_plusplus = use_plusplus
        self.n_init = n_init
        self.rng = np.random.RandomState(self.random_state)

    def fit(self, X: ndarray, y=None):
        if self.use_plusplus:
            seeds = self.rng.randint(
                np.iinfo(np.int32).max, size=self.n_init)
            labels = []
            inertias = []
            models = []
            for seed in seeds:
                if self.verbose > 0:
                    print(f'Kmeans++ initialization with seed {seed}')

                self.initial_centroids_ = np.array(
                        kmeans_plusplus_clusters_initialization(
                        X, self.distance, self.n_clusters, seed,
                        self.n_jobs, self.verbose)
                    )
                self.models.append(Kmeans(n_clusters=self.n_clusters,
                                    init=self.initial_centroids_,
                                    n_jobs=self.n_jobs,
                                    random_state=seed,
                                    max_iter=self.max_iter,
                                    tol=self.tol,
                                    metric=self.metric).fit(X))
                self.labels_.append(self.models[-1].labels_)
                self.inertias_.append(self.models[-1].inertia_)

                if self.verbose > 0:
                    print(f'Inertia: {self.inertias_[-1]}')
                    print("-"*120 + '\n')

            best = np.argmin(self.inertias_)
            self.kmeans_ = self.models[best]
            self.labels_ = self.labels_[best]
            self.inertia_ = self.inertias_[best]

        else:
            self.kmeans_ = Kmeans(n_clusters=self.n_clusters,
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        n_init=self.n_init,
                        metric=self.metric).fit(X)

        return self

    def transform(self, X: ndarray):
        return self.kmeans_.labels_

    def fit_transform(self, X: ndarray, y=None):
        return self.fit(X).transform(X)



class LabelsToImage(BaseEstimator, TransformerMixin):
    """A class to transform labels results of Kmeans++ to an image.
    """

    def __init__(self, height: int, width: int,
                 window_size: int, overlap: int = 0):
        assert window_size % 2 == 1, 'Window size must be odd.'
        assert overlap >= 0, 'Overlap must be positive.'
        assert overlap <= window_size//2, 'Overlap must be smaller or equal than int(window_size/2).'
        self.height = height
        self.width = width
        self.overlap = overlap
        self.window_size = window_size

    def fit(self, X: ndarray, y=None):
        return self

    def transform(self, X: ndarray):
        # Compute reshape size thanks ot window-size before overlap
        height = self.height - self.window_size + 1
        width = self.width - self.window_size + 1
        # Taking into account overlap
        if self.overlap > 0:
            height = ceil(height/self.overlap) 
            width = ceil(width/self.overlap)

        # Reshape to (height, weight)
        return X.reshape((height, width))

    def fit_transform(self, X: ndarray, y=None):
        return self.fit(X).transform(X)


# Functions
# ---------
def pca_image(image, nb_components):
    """ A function that centers data and applies PCA on an image.
        Inputs:
            * image: numpy array of the image.
            * nb_components: number of components to keep.
        FROM: Antoine Collas
    """
    # center pixels
    h, w, p = image.shape
    X = image.reshape((h*w, p))
    mean = np.mean(X, axis=0)
    image = image - mean
    X = X - mean
    # check pixels are centered
    assert (np.abs(np.mean(X, axis=0)) < 1e-8).all()

    # apply PCA
    SCM = (1/len(X))*X.conj().T@X
    d, Q = la.eigh(SCM)
    reverse_idx = np.arange(len(d)-1, -1, step=-1)
    Q = Q[:, reverse_idx]
    Q = Q[:, :nb_components]
    image = image@Q

    return image


def compute_distances_parallel(center: ndarray, X: ndarray,
                               distance: Callable,
                               n_jobs: int = 1,
                               **kwself) -> List[float]:
    """Compute distances between samples in parallel.

    Parameters
        center: ndarray
            Center of the distances.
        
        X: ndarray
            Data array of shape (n_samples, n_features).

        distance: Callable
            Distance function between two samples.

        n_jobs: int
            Number of jobs to run in parallel.

        **kwself: dict
            Additional arguments for distance function.

    Returns
        distances: List[float]
            List of distances.
    """
    if n_jobs == 1:
        return [distance(x, center, **kwself) for x in X]
    else:
        return Parallel(n_jobs=n_jobs)(
                delayed(distance)(x, center, **kwself) for x in X)


def kmeans_plusplus_clusters_initialization(X: ndarray, distance: Callable,
                                            n_clusters: int,
                                            random_state: int,
                                            n_jobs: int = 1,
                                            verbose: int = 0,
                                            **kwself) ->\
                                                    List[ndarray]:
    """
    Kmeans++ initialization of clusters centers.

    Parameters
        X: ndarray
            Data array of shape (n_samples, n_features).

        distance: Callable
            Distance function between two samples.

        n_clusters: int
            Number of clusters.

        random_state: int
            Random state for reproducibility.
            

        n_jobs: int
            Number of jobs to run in parallel.

        verbose: int
            Verbosity level.

        **kwself: dict
            Additional arguments for distance function.

    Returns
        centroids: List[ndarray]
            List of clusters centers.
    """
    rng = np.random.RandomState(random_state)
    centroids = []
    indexes = []

    # Choose first centroid randomly
    if verbose > 0:
        logging.info('Choosing first centroid randomly...')
    idx = rng.randint(0, X.shape[0])
    centroids.append(X[idx])
    indexes.append(idx)
    if verbose > 1:
        logging.info(f'First centroid {idx}: {centroids[-1]}')

    if verbose > 0:
        logging.info('Choosing next centroids...')
        pbar = tqdm(total=n_clusters-1)

    for _ in range(1, n_clusters):
        # Compute distances to centroids
        distances = compute_distances_parallel(
                centroids[-1], X, distance, n_jobs, **kwself)
        # Choose next centroid randomly with probability proportional to distance
        idx = rng.choice(np.arange(X.shape[0]), p=distances/np.sum(distances))
        while idx in indexes:
            idx = rng.choice(np.arange(X.shape[0]), p=distances/np.sum(distances))
        centroids.append(X[idx])
        indexes.append(idx)

        if verbose > 0:
            pbar.update(1)
            if verbose > 1:
                logging.info(f'Next centroid {idx}: {centroids[-1]}')

    if verbose:
        pbar.close()

    return centroids

def _get_classes(C, gt):
    classes_C = np.unique(C).astype(int)
    classes_gt = np.unique(gt).astype(int)
    classes = np.array(list(set().union(classes_C, classes_gt)))
    classes = classes.astype(np.int64)
    classes = classes[classes >= 0]
    classes = np.sort(classes)
    return classes

def assign_segmentation_classes_to_gt_classes(C, gt, normalize=False):
    """ A function that assigns the classes of the segmentation to
        the classes of the ground truth.
        BE CAREFUL : negative classes are always ignored, both in C and gt.
        Inputs:
            * C: segmented image.
            * gt: ground truth.
            * normalize: normalize each row of the cost matrix.
        Ouput:
            * segmented image with the right classes.
    """
    # get classes
    classes_gt = _get_classes(gt, gt)
    classes_C = _get_classes(C, C)
    assert len(classes_gt) >= len(classes_C)

    cost_matrix = np.zeros((len(classes_gt), len(classes_C)))

    for i, class_gt in enumerate(classes_gt):
        mask = (gt == class_gt)
        if normalize:
            nb_pixels = np.sum(mask)
        for j, class_C in enumerate(classes_C):
            cost = -np.sum(C[mask] == class_C)
            if normalize:
                cost /= nb_pixels
            cost_matrix[i, j] = cost

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    row_ind = classes_gt[row_ind]
    col_ind = classes_C[col_ind]
    new_C = C.copy()
    for i, j in zip(col_ind, row_ind):
        new_C[C == i] = j

    return new_C

def compute_mIoU(C, gt):
    """ A function that computes the mean of Intersection over Union between
    a segmented image (c) and a ground truth (gt).
    BE CAREFUL, negative values in gt are considered
    as no annotation available.
        Inputs:
            * C: segmented image.
            * gt: ground truth.
            * classes: list of classes used to compute the mIOU
        Ouputs:
            * IoU, mIOU
    """
    # get classes
    classes = _get_classes(C, gt)

    IoU = list()
    for i in classes:
        inter = np.sum((C == i) & (gt == i))
        union = np.sum(((C == i) | (gt == i)) & np.isin(gt, classes))
        IoU.append(inter/union)
    mIoU = np.mean(IoU)
    return IoU, mIoU

