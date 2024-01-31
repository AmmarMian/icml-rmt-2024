import numpy as np

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, ClusterMixin
from sklearn.utils.extmath import softmax

from .mean import RMT_geometric_mean, geometric_mean
from .distance import RMT_squaredFisherDistance_deterministic_SCMs
from .covariance import SCM_estimator


from tqdm import tqdm

class MDM_RMT(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Classification by Minimum Distance to Mean with RMT based Fisher distance estimator.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated according to the RMT based Fisher distance estimator. 
    Then, for each new point, the class is affected according to the nearest centroid.

    Parameters
    ----------
    n_jobs : int, default=1
        Number of jobs.
    n_degrees_of_freedom : int, default=None
        Estimated actual number of independant samples in the trials.
        If None, the number of samples of the trials is taken.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.
    covmeans_ : list of ``n_classes`` ndarrays of shape (n_channels, n_channels)
        Centroids for each class.
    """

    def __init__(self, n_degrees_of_freedom=None, n_jobs=1):
        self.n_jobs = n_jobs
        self.n_dof = n_degrees_of_freedom
        self.n_degrees_of_freedom = n_degrees_of_freedom

    def fit(self, X, y):
        """
        Fit (estimate) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Set of trials.
        y : ndarray, shape (n_matrices,)
            Labels for each trial.

        Returns
        -------
        self : MDM_RMT instance
            The MDM_RMT instance.
        """
        self.classes_ = np.unique(y)
        X_ = X.swapaxes(-2,-1)
        if self.n_jobs == 1:
            self.covmeans_ = [
                RMT_geometric_mean(X_[y == ll], init=geometric_mean(SCM_estimator(X_[y == ll]), max_iterations=20), n_dof=self.n_dof, max_iterations=50)
                for ll in self.classes_]
        else:
            self.covmeans_ = Parallel(n_jobs=self.n_jobs)(
                delayed(RMT_geometric_mean)(X_[y == ll], init=geometric_mean(SCM_estimator(X_[y == ll]), max_iterations=20), n_dof=self.n_dof, max_iterations=50)
                for ll in self.classes_)

        return self

    def _predict_squared_distances(self, X):
        """Helper to predict the distance. Equivalent to transform."""
        n_centroids = len(self.covmeans_)

        if self.n_dof is None:
            n_dof = X.shape[-1]
        else:
            n_dof = self.n_dof
        SCMs = SCM_estimator(X.swapaxes(-2,-1))
        if self.n_jobs == 1:
            dist2 = [RMT_squaredFisherDistance_deterministic_SCMs(self.covmeans_[m], SCMs, n_dof)
                    for m in range(n_centroids)]
        else:
            dist2 = Parallel(n_jobs=self.n_jobs)(delayed(RMT_squaredFisherDistance_deterministic_SCMs)(
                self.covmeans_[m], SCMs, n_dof)
                for m in range(n_centroids))

        dist2 = np.array(dist2).T
        return dist2

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Set of trials.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each trial according to the closest centroid.
        """
        dist2 = self._predict_squared_distances(X)
        return self.classes_[dist2.argmin(axis=1)]

    def transform(self, X):
        """Get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Set of trials.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            The square of the RMT corrected distance to each centroid.
        """
        return self._predict_squared_distances(X)

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """Predict proba using softmax of negative squared distances.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_matrices, n_classes)
            Probabilities for each class.
        """
        return softmax(-self._predict_squared_distances(X))

    
        
class KMeans_RMT(BaseEstimator, ClusterMixin):
    """RMT based KMeans clustering.
    """

    def __init__(self, n_clusters=16, n_jobs=1, tol=1e-3, max_iter=100,
                n_init=10, random_state: int = 42, verbose=0):
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        self.tol = tol
        self.max_iter = max_iter
        self.MDM = [MDM_RMT(n_jobs=self.n_jobs) for _ in range(n_init)]
        self.verbose = verbose
        self.random_state = random_state
        self.n_init = n_init
        self.rng = np.random.RandomState(self.random_state)
        self.fitted = False


    def _init_random_labels(self, X, seed):
        """Initialize the labels randomly."""
        rng = np.random.RandomState(seed)
        n_samples = X.shape[0]
        labels = rng.randint(self.n_clusters, size=n_samples)
        steps = 0
        while len(np.unique(labels)) < self.n_clusters:
            labels = self._init_random_labels(X, seed+self.n_init+steps)
            steps += 1
        return labels

    def fit(self, X, y=None):

        seeds = self.rng.randint(
            np.iinfo(np.int32).max, size=self.n_init)
        labels_list = []
        inertias_list = []
        for i, seed in enumerate(seeds):

            # Kmeans one init
            # -------------------------------------------------------------
            if self.verbose > 0:
                print(f'KMeans_RMT: init {i+1}/{self.n_init}')

            # Initialize labels randomly
            labels = self._init_random_labels(X, seed)

            # Compute initial centroids with RMT-MDM
            self.MDM[i].fit(X, labels)

            # Iterate until convergence
            delta = np.inf
            n_iter = 0

            if self.verbose > 0:
                p_bar = tqdm(total=self.max_iter, desc='KMeans_RMT', leave=True)

            while delta > self.tol and n_iter < self.max_iter:

                # # Assign each sample to the closest centroid
                # labels_new = self.MDM.predict(X)

                # # Compute new centroids
                # self.MDM.fit(X, labels_new)

                # Compute distance to centroids
                dist2 = self.MDM[i].transform(X)

                # Assign each sample to the closest centroid
                labels_new = np.argmin(dist2, axis=1)

                # Compute new centroids
                self.MDM[i].fit(X, labels_new)

                # Compute delta
                delta = np.sum(labels_new != labels)/len(labels)
                labels = labels_new
                n_iter += 1

                if self.verbose > 0:
                    p_bar.update(1)
                    p_bar.set_description(f'KMeans_RMT (delta={delta})')

            # compute inertia
            dist2 = self.MDM[i].transform(X)
            inertia = np.sum([np.sum(dist2[labels==i, i])
                            for i in range(len(self.MDM[i].covmeans_))])
            if self.verbose > 0:
                print(f'KMeans_RMT: init {i+1}/{self.n_init} - inertia: {inertia:.2f}')
            labels_list.append(labels)
            inertias_list.append(inertia)
            if self.verbose > 0:
                print('-'*120+'\n')
            # ------------------------------------------------------------------------------------

        # Choose best init
        if self.verbose > 0:
            print('KMeans_RMT: choosing best init...')
            print(f'inertias: {inertias_list}')
        best_init = np.argmin(inertias_list)
        labels = labels_list[best_init]
        inertia = inertias_list[best_init]
        self.inertia_ = inertia
        self.MDM = self.MDM[best_init]
        self.labels_ = labels
        self.fitted = True

        return self

    def predict(self, X):
        return self.MDM.predict(X)

    def transform(self, X, y=None):
        if not self.fitted:
            self.fit(X)
        return self.labels_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
