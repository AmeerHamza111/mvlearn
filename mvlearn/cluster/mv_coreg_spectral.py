
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
# Implements multi-view spectral clustering algorithm for data with
# multiple views.


import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from ..utils.utils import check_Xs
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.neighbors import NearestNeighbors

AFFINITY_METRICS = ['rbf', 'nearest_neighbors', 'poly']


class MultiviewCoRegSpectralClustering(BaseEstimator):

    r'''
    An implementation of multi-view spectral clustering using the
    basic co-training framework as described in [#1Clu]_.
    This algorithm can handle 2 or more views of data.

    Parameters
    ----------
    n_clusters : int
        The number of clusters

    n_views : int, optional, default=2
        The number of different views of data.

    random_state : int, optional, default=None
        Determines random number generation for k-means.

    info_view : int, optional, default=None
        The most informative view. Must be between 0 and n_views-1
        If given, then the final clustering will be performed on the
        designated view alone. Otherwise, the algorithm will concatenate
        across all views and cluster on the result.

    max_iter : int, optional, default=10
        The maximum number of iterations to run the clustering
        algorithm.

    n_init : int, optional, default=10
        The number of random initializations to use for k-means clustering.

    affinity : string, optional, default='rbf'
        The affinity metric used to construct the affinity matrix. Options
        include 'rbf' (radial basis function), 'nearest_neighbors', and
        'poly' (polynomial)

    gamma : float, optional, default=None
        Kernel coefficient for rbf and polynomial kernels. If None then
        gamma is computed as 1 / (2 * median(pair_wise_distances(X))^2)
        for each data view X.

    n_neighbors : int, optional, default=10
        Only used if nearest neighbors is selected for affinity. The
        number of neighbors to use for the nearest neighbors kernel.

    Notes
    -----


    References
    ----------
    .. [#1Clu] Abhishek Kumar and Hal Daume. A Co-training Approach for
            Multiview Spectral Clustering. In International Conference
            on Machine Learning, 2011
    '''
    def __init__(self, n_clusters=2, n_views=2, v_lambda=2, random_state=None,
                 info_view=None, max_iter=10, n_init=10, affinity='rbf',
                 gamma=None, n_neighbors=10):

        super().__init__()

        if not (isinstance(n_clusters, int) and n_clusters > 0):
            msg = 'n_clusters must be a positive integer'
            raise ValueError(msg)

        if not (isinstance(n_views, int) and n_views > 1):
            msg = 'n_views must be a positive integer greater than 1'
            raise ValueError(msg)

        if random_state is not None:
            msg = 'random_state must be convertible to 32 bit unsigned integer'
            try:
                random_state = int(random_state)
            except ValueError:
                raise ValueError(msg)
            np.random.seed(random_state)

        self.info_view = None
        if info_view is not None:
            if (isinstance(info_view, int)
                    and (info_view >= 0 and info_view < n_views)):
                self.info_view = info_view
            else:
                msg = 'info_view must be an integer between 0 and n_clusters-1'
                raise ValueError(msg)

        if not (isinstance(max_iter, int) and (max_iter > 0)):
            msg = 'max_iter must be a positive integer'
            raise ValueError(msg)

        if not (isinstance(n_init, int) and n_init > 0):
            msg = 'n_init must be a positive integer'
            raise ValueError(msg)

        if affinity not in AFFINITY_METRICS:
            msg = 'affinity must be a valid affinity metric'
            raise ValueError(msg)

        if gamma is not None and not ((isinstance(gamma, float) or
                                       isinstance(gamma, int)) and gamma > 0):
            msg = 'gamma must be a positive float'
            raise ValueError(msg)

        if not (isinstance(n_neighbors, int) and n_neighbors > 0):
            msg = 'n_neighbors must be a positive integer'
            raise ValueError(msg)

        # Need to check if lambda is a valid value

        self.n_clusters = n_clusters
        self.n_views = n_views
        self.random_state = random_state
        self.info_view = info_view
        self.max_iter = max_iter
        self.n_init = n_init
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.v_lambda = v_lambda
        self._objective = None
        self._embedding = None
        
    def _affinity_mat(self, X):

        r'''
        Computes the affinity matrix based on the selected
        kernel type.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data matrix from which we will compute the
            affinity matrix.

        Returns
        -------
        sims : array-like, shape (n_samples, n_samples)
            The resulting affinity kernel.

        '''

        sims = None

        # If gamma is None, then compute default gamma value for this view
        gamma = self.gamma
        if self.gamma is None:
            distances = cdist(X, X)
            gamma = 1 / (2 * np.median(distances) ** 2)
        # Produce the affinity matrix based on the selected kernel type
        if (self.affinity == 'rbf'):
            sims = rbf_kernel(X, gamma=gamma)
        elif(self.affinity == 'nearest_neighbors'):
            neighbor = NearestNeighbors(n_neighbors=self.n_neighbors)
            neighbor.fit(X)
            sims = neighbor.kneighbors_graph(X).toarray()
        else:
            sims = polynomial_kernel(X, gamma=gamma)

        return sims

    def _init_umat(self, X):

        r'''
        Computes the top several eigenvectors of the
        normalized graph laplacian of a given similarity matrix.
        The number of eigenvectors returned is equal to n_clusters.

        Parameters
        ----------
        X : array-like, shape(n_samples, n_samples)
            The similarity matrix for the data in a single view.

        Returns
        -------
        la_eigs : array-like, shape(n_samples, n_clusters)
            The top n_cluster eigenvectors of the normalized graph
            laplacian.
        '''

        # Compute the normalized laplacian
        d_mat = np.diag(np.sum(X, axis=1))
        # Check abs in the other spectral clustering algo
        d_alt = np.sqrt(np.linalg.inv(d_mat))
        laplacian = d_alt @ X @ d_alt

        # Make the resulting matrix symmetric
        laplacian = (laplacian + np.transpose(laplacian)) / 2.0
        # Obtain the top n_cluster eigenvectors of the laplacian
        # u_mat, d_mat, _ = sp.sparse.linalg.svds(laplacian, k=self.n_clusters)
        e_vals, e_vecs = np.linalg.eig(laplacian)
        obj_val = np.sum(e_vals[:self.n_clusters])
        u_mat = np.real(e_vecs[:, :self.n_clusters])
        return u_mat, laplacian, obj_val

    def _compute_eigs(self, X):
        e_vals, e_vecs = np.linalg.eig(X)
        obj_val = np.sum(e_vals[:self.n_clusters])
        u_mat = np.real(e_vecs[:, :self.n_clusters])
        return u_mat, obj_val

    def fit_predict(self, Xs):

        r'''
        Performs clustering on the multiple views of data and returns
        the cluster labels.

        Parameters
        ----------

        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

            This list must be of size n_views, corresponding to the number
            of views of data. Each view can have a different number of
            features, but they must have the same number of samples.

        Returns
        -------
        predictions : array-like, shape (n_samples,)
            The predicted cluster labels for each sample.
        '''

        Xs = check_Xs(Xs)
        if len(Xs) != self.n_views:
            msg = 'Length of Xs must be the same as n_views'
            raise ValueError(msg)

        # Compute the similarity matrices
        sims = [self._affinity_mat(X) for X in Xs]
        
        # Initialize matrices of eigenvectors
        U_mats = []
        L_mats = []
        obj_vals = np.zeros((self.n_views, self.max_iter))
        for ind in range(len(sims)):
            u_mat, l_mat, o_val = self._init_umat(sims[ind])
            U_mats.append(u_mat)
            L_mats.append(l_mat)
            obj_vals[ind, 0] = o_val
            
        # Perform clustering for the first view
        U_1 = U_mats[0]
        normed_1 = np.sqrt(np.diag(U_1 @ U_1.T))
        normed_1[normed_1 == 0.0] = 1
        U_1 = np.linalg.inv(np.diag(normed_1)) @ U_1
        print(U_1[:6])
        
        # Now iteratively solve for all U's
        n_items = Xs[0].shape[0]
        for it in range(1, self.max_iter):
            for v1 in range(1, self.n_views):
                l_comp = np.zeros((n_items, n_items))
                for v2 in range(self.n_views):
                    if v1 != v2:
                        l_comp = l_comp + U_mats[v2] @ U_mats[v2].T
                l_comp = (l_comp + l_comp.T) / 2
                l_mat = L_mats[v1] + self.v_lambda * l_comp
                #U_mats[v1], d_mat, _ = sp.sparse.linalg.svds(l_mat, k=self.n_clusters)
                U_mats[v1], obj_vals[v1, it] = self._compute_eigs(l_mat)

            l_comp = np.zeros((n_items, n_items))
            for vi in range(self.n_views):
                if vi != 0:
                    l_comp = l_comp + U_mats[vi] @ U_mats[vi].T
            l_comp = (l_comp + l_comp.T) / 2
            l_mat = L_mats[0] + self.v_lambda * l_comp
            U_mats[0], obj_vals[0, it] = self._compute_eigs(l_mat)
            print(U_mats[0])
            #U_mats[0], d_mat, _ = sp.sparse.linalg.svds(l_mat, k=self.n_clusters)
            #obj_vals[0, it] = np.sum(d_mat)

        self._objective = obj_vals
        
        U_norm = list()
        for vi in range(self.n_views):
            normed_v = np.sqrt(U_mats[vi] @ U_mats[vi].T)
            normed_v[normed_v == 0] = 1
            U_norm.append(np.linalg.inv(np.diag(np.diag(normed_v))) @ U_mats[vi])

            
        #V_mat = np.hstack(U_norm)
        V_mat = np.hstack(U_mats)
        norm_v = np.sqrt(np.diag(V_mat @ V_mat.T))
        norm_v[norm_v == 0] = 1
        self._embedding = np.linalg.inv(np.diag(norm_v)) @ V_mat
        
        kmeans = KMeans(n_clusters=self.n_clusters,
                        random_state=self.random_state)
        predictions = kmeans.fit_predict(V_mat)
        
        return predictions
