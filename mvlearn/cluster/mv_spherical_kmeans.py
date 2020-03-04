# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Implements multi-view kmeans clustering algorithm for data with 2-views.

from .mv_kmeans import MultiviewKMeans
import numpy as np
from ..utils.utils import check_Xs
from sklearn.preprocessing import normalize


class MultiviewSphericalKMeans(MultiviewKMeans):

    r'''
    An implementation of Multi-View Spherical K-Means using the
    co-EM framework as described in [#1Clu]_. This algorithm is
    most suitable for cases in which the different views of data
    are conditionally independent. This algorithm currently handles
    two views of data.

    Parameters
    ----------
    n_clusters : int, optional, default=2
        The number of clusters

    random_state : int, optional, default=None
        Determines random number generation for initializing centroids.
        Can seed the random number generator with an int.

    patience : int, optional, default=5
        The number of EM iterations with no decrease in the objective
        function after which the algorithm will terminate.

    max_iter : int, optional, default=None
        The maximum number of EM iterations to run before
        termination.

    n_init : int, optional, default=5
        Number of times the k-means algorithm will run on different
        centroid seeds. The final result will be the best output of
        n_init runs with respect to total inertia across all views.

    init : {'k-means++', 'random'} or list of array-likes, default='k-means++'
        Method of initializing centroids.

        'k-means++': selects initial cluster centers for k-means clustering
        via a method that speeds up convergence.

        'random': choose n_cluster samples from the data for the initial
        centroids.

        If a list of array-likes is passed, the list should have a length of
        equal to the number of views. Each of the array-likes should have
        the shape (n_clusters, n_features_i) for the ith view, where
        n_features_i is the number of features in the ith view of the input
        data.

    Attributes
    ----------
    centroids_ : list of array-likes
        - centroids_ length: n_views
        - centroids_[i] shape: (n_clusters, n_features_i)

        The cluster centroids for each of the two views. centroids_[0]
        corresponds to the centroids of view 1 and centroids_[1] corresponds
        to the centroids of view 2.


    Notes
    -----

    Multi-view Spherical KMeans clustering adapts the traditional spherical
    kmeans clustering algorithm to handle two views of data. This algorithm
    is similar to the Mult-view KMeans algorithm, except it uses cosine
    distance instead of euclidean distance for the purposes of computing
    the optimization objective and making assignments. This algorithm
    requires that a conditional independence assumption between views holds
    true. In cases where both views are informative and conditionally
    independent, Multi-view Spherical KMeans clustering can outperform its
    single-view analog run on a concatenated version of the two views of data.
    This is quite useful for applications where you wish to cluster data from
    two different modalities or data with features that naturally fall into two
    different partitions. Multi-view Spherical KMeans works by iteratively 
    performing the maximization and expectation steps of traditional EM in 
    one view, and then using the computed hidden variables as the input for the
    maximization step in the other view. This algorithm, referred to as Co-EM, 
    is described below.

    *Co-EM Algorithm*

    Input: Unlabeled data D with 2 views

        #. Initialize :math:`\Theta_0^{(2)}`, T, :math:`t = 0`.

        #. E step for view 2: compute expectation for hidden variables given

        #. Loop until stopping criterion is true:

            #. For v = 1 ... 2:

                #. :math:`t = t + 1`

                #. M step view v: Find model parameters :math:`\Theta_t^{(v)}`
                   that maximize the likelihood for the data given the expected
                   values for hidden variables of view :math:`\overline{v}` of
                   iteration :math:`t` - 1

                #. E step view :math:`v`: compute expectation for hidden
                   variables given the model parameters :math:`\Theta_t^{(v)}`

        #. return combined :math:`\hat{\Theta} = \Theta_{t-1}^{(1)} \cup
           \Theta_t^{(2)}`

    The final assignment of examples to partitions is performed by assigning
    each example to the cluster with the largest averaged posterior
    probability over both views.

    References
    ----------
    .. [#1Clu] Bickel S, Scheffer T (2004) Multi-view clustering. Proceedings
            of the 4th IEEE International Conference on Data Mining, pp. 19–26
    '''

    def __init__(self, n_clusters=2, random_state=None, init='k-means++',
                 patience=5, max_iter=None, n_init=5):

        super().__init__(n_clusters=n_clusters, random_state=random_state,
                         init=init, patience=patience, max_iter=max_iter,
                         n_init=n_init)

    def _compute_dist(self, X, Y):

        r'''
        Function that computes the pairwise distance between each row of X
        and each row of Y. The distance metric used here is Cosine
        distance.

        Parameters
        ----------
        X: array-like, shape (n_samples_i, n_features)
            An array of samples.
        Y: array-like, shape (n_samples_j, n_features)
            Another array of samples. Second dimension is the same size
            as the second dimension of X.

        Returns
        -------
        distances: array-like, shape (n_samples_i, n_samples_j)
            An array containing the pairwise distances between each
            row of X and each row of Y.
        '''

        cosine_dist = 1 - X @ np.transpose(Y)
        return cosine_dist

    def _init_centroids(self, Xs):

        r'''
        Initializes the centroids for Multi-view KMeans or KMeans++ depending
        on which has been selected.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

            This list must be of size 2, corresponding to the two views of
            the data. The two views can each have a different number of
            features, but they must have the same number of samples.

        Returns
        -------
        centroids : list of array-likes
            - centroids length: n_views
            - centroids[i] shape: (n_clusters, n_features_i)

            The cluster centroids for each of the two views. centroids[0]
            corresponds to the centroids of view 1 and centroids[1] corresponds
            to the centroids of view 2. These are not yet the final cluster
            centroids.
        '''
        centroids = None
        if self.init == 'random':
            # Random initialization of centroids
            indices = np.random.choice(Xs[0].shape[0], self.n_clusters)
            centers1 = Xs[0][indices]
            centers2 = Xs[1][indices]
            centroids = [centers1, centers2]
        elif self.init == 'k-means++':
            # Initializing centroids via kmeans++ implementation
            indices = list()
            centers2 = list()
            indices.append(np.random.randint(Xs[1].shape[0]))
            centers2.append(Xs[1][indices[0], :])

            # Compute the remaining n_cluster centroids
            for cent in range(self.n_clusters - 1):
                dists = self._compute_dist(centers2, Xs[1])
                dists = np.min(dists, axis=1)
                max_index = np.argmax(dists)
                indices.append(max_index)
                centers2.append(Xs[1][max_index])

            centers1 = Xs[0][indices]
            centers2 = np.array(centers2)
            centroids = [centers1, centers2]
        else:
            # Somewhere here, check that centroids are normalized
            centroids = self.init
            try:
                centroids = check_Xs(centroids, enforce_views=2)
                if (len(centroids) != 2):
                    msg = 'Number of views for the centroids must be 2'
                    raise ValueError(msg)
            except ValueError:
                msg = 'init must be a valid centroid initialization'
                raise ValueError(msg)
            for ind in range(len(centroids)):
                if centroids[ind].shape[0] != self.n_clusters:
                    msg = 'number of centroids per view must equal n_clusters'
                    raise ValueError(msg)
                if centroids[ind].shape[1] != Xs[ind].shape[1]:
                    msg = ('feature dimensions of cluster centroids'
                           + ' must match those of data')
                    raise ValueError(msg)

        return centroids

    def _em_step(self, X, partition, centroids):

        r'''
        A function that computes one iteration of expectation-maximization.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            An array of samples representing a single view of the data.

        partition: array-like, shape (n_samples,)
            An array of cluster labels indicating the cluster to which
            each data sample is assigned. This is essentially a partition
            of the data points.

        centroids: array-like, shape (n_clusters, n_features)
            The current cluster centers.

        Returns
        -------
        new_parts: array-like, shape (n_samples,)
            The new cluster assignments for each sample in the data after
            the data has been repartitioned with respect to the new
            cluster centers.

        new_centers: array-like, shape (n_clusters, n_features)
            The updated cluster centers.

        o_funct: float
            The new value of the objective function.
        '''

        n_samples = X.shape[0]
        new_centers = list()
        for cl in range(self.n_clusters):
            # Recompute centroids using samples from each cluster
            mask = (partition == cl)
            if (np.sum(mask) == 0):
                new_centers.append(centroids[cl])
            else:
                cent = np.sum(X[mask], axis=0)
                new_centers.append(cent)
        new_centers = np.vstack(new_centers)
        new_centers = normalize(new_centers)

        # Compute expectation and objective function
        distances = self._compute_dist(X, new_centers)
        new_parts = np.argmin(distances, axis=1).flatten()
        min_dists = distances[np.arange(n_samples), new_parts]
        o_funct = np.sum(min_dists)
        return new_parts, new_centers, o_funct

    def _preprocess_data(self, Xs):

        r'''
        Checks that the inputted data is in the correct format and
        normalizes to make row vectors unit length.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

            This list must be of size 2, corresponding to the two views of
            the data. The two views can each have a different number of
            features, but they must have the same number of samples.

        Returns
        -------
        Xs_new : list of array-likes
            - centroids length: n_views
            - centroids[i] shape: (n_clusters, n_features_i)

            The data samples after they have been checked and normalized.
        '''

        # Check that the input data is valid
        Xs = check_Xs(Xs, enforce_views=2).copy()
        if (len(Xs) != 2):
            msg = 'Number of views of data must be 2'
            raise ValueError(msg)

        # Normalize the input samples
        for view in range(len(Xs)):
            Xs[view] = Xs[view].astype(float)
            Xs[view] = normalize(Xs[view])

        return Xs

    def fit(self, Xs):

        r'''
        Fit the cluster centroids to the data.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

            This list must be of size 2, corresponding to the two views of
            the data. The two views can each have a different number of
            features, but they must have the same number of samples.

        Returns
        -------
        self : returns an instance of self.
        '''

        Xs = self._preprocess_data(Xs)
        super().fit(Xs)
        # Normalize the centroids
        for view in range(len(self.centroids_)):
            self.centroids_[view] = normalize(self.centroids_[view])