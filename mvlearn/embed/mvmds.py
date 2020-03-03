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

from mvlearn.embed.base import BaseEmbed
from mvlearn.utils.utils import check_Xs
import warnings
import numpy as np
from sklearn.metrics import euclidean_distances


class MVMDS(BaseEmbed):

    r"""
    An implementation of Classical Multiview Multidimensional Scaling for
    jointly reducing the dimensions of multiple views of data.

    A Euclidean distance matrix is created for each view, double centered,
    and the k largest common eigenvectors between the matrices are returned
    based on the stepwise estimation of common principal components.

    Parameters
    ----------
    n_components : int (positive), default=None
        Represents the number of components that the user would like to
        be returned from the algorithm. This value must be greater than
        0 and less than the number of samples within each view.

    num_iter: int (positive), default=15
        Number of iterations stepwise estimation goes through. Detailed
        in Trendafilov paper.

    Attributes
    ----------
    components: numpy.ndarray
            - components shape: (n_samples, n_components)
            MVMDS components of Xs

    References
    ----------
    .. [#1] Trendafilov, Nickolay T. “Stepwise Estimation of Common Principal
            Components.” Computational Statistics &amp; Data Analysis, vol. 54,
            no. 12, 2010, pp. 3446–3457., doi:10.1016/j.csda.2010.03.010.
    """

    def __init__(self, n_components=None, num_iter=15):

        super().__init__()
        self.components = None
        self.n_components = n_components
        self.num_iter = num_iter

    def _commonpcs(self, Xs):

        r"""
        Finds Stepwise Estimation of Common Principal Components as described
        by common Trendafilov implementations based on the following paper:

        https://www.sciencedirect.com/science/article/pii/S016794731000112X

        Parameters
        ----------
        Xs: List of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

        Returns
        -------
        components: numpy.ndarray
            - components shape: (n_samples, n_components)
            MVMDS components of Xs

        Notes
        -----

        The Common Principal Component (CPC) model states that given :math:'k'
        normal populations it is assumed that their :math:'p x p' covariance
        matrices, :math:'$\Sigma_i, i = 1,2,...,k$' can be simultaneously
        decomposed in the form:

        ..math::

            $$\Sigma_i = QD_i^2Q^T$$

        where :math:'Q' is a common :math:'p x p' matrix that is orthogonal to
        all :math:'$\Sigma_i$' and :math:'D_i^2' is a positive :math:'p x p'
        diagonal matrix.

        The matrix :math:'Q' contains all of the CPCs. The CPCs are found in a
        step-wise process. :math:'S_i' represents the :math:'i-th' sample
        :math:'p x p' covariance matrix. To find the CPCs we want to solve p
        identical minimization problems

        ..math::

            $$Minimize \sum_{i=1}^{k}n_ilog(q^TS_iq) $$
            \newline
            $$Subject to q^Tq = 1$$

        where :math:'n_i' represents the degrees of freedom in the :math:'i-th'
        population. This allows us to find the first CPC, :math:'q_p.'

        The next CPC can be found through the same equation:

        ..math::

            $$Minimize \sum_{i=1}^{k}n_ilog(q^TS_iq) $$
            \newline
            $$Subject to q^Tq = 1$$

        where this problem is also subject to :math:'q_{p-1}' being orthogonal to
        :math:'q_p'. This process continues until all CPCs are created that make
        up :math:'Q'.

        """
        n = p = Xs.shape[1]

        views = len(Xs)

        n_num = np.array([n] * views)/np.sum(np.array([n] * views))

        components = np.zeros((p, self.n_components))

        # Initialized by paper
        pi = np.eye(p)

        s = np.zeros((p, p))

        for i in np.arange(views):
            s = s + (n_num[i] * Xs[i])

        e1, e2 = np.linalg.eigh(s)

        # Orders the eigenvalues
        q0 = e2[:, ::-1]

        for i in np.arange(self.n_components):

            # Each q is a particular eigenvalue
            q = q0[:, i]
            q = np.array(q).reshape(len(q), 1)
            d = np.zeros((1, views))

            for j in np.arange(views):

                # Represents mu from the paper.
                d[:, j] = np.dot(np.dot(q.T, Xs[j]), q)

            # stepwise iterations
            for j in np.arange(self.num_iter):
                s2 = np.zeros((p, p))

                for yy in np.arange(views):
                    d2 = n_num[yy] * np.sum(np.array([n] * views))

                    # Dividing by .0001 is to prevent divide by 0 error
                    if d[:, yy] == 0:
                        s2 = s2 + (d2 * Xs[yy] / .0001)

                    else:
                        # Refers to d value from previous iteration
                        s2 = s2 + (d2 * Xs[yy] / d[:, yy])

                # eigenvectors dotted with S matrix and pi
                w = np.dot(s2, q)

                w = np.dot(pi, w)

                q = w / np.sqrt(np.dot(w.T, w))

                for yy in np.arange(views):

                    d[:, yy] = np.dot(np.dot(q.T, Xs[yy]), q)

            # creates next component
            components[:, i] = q[:, 0]
            # initializes pi for next iteration
            pi = pi - np.dot(q, q.T)

        return(components)

    def fit(self, Xs):

        """
        Calculates dimensionally reduced components by inputting the Euclidean
        distances of each view, double centering them, and using the _commonpcs
        function to find common components between views. Works similarly to
        traditional, single-view Multidimensional Scaling.

        Parameters
        ----------

        Xs: list of array-likes or numpy.ndarray
                - Xs length: n_views
                - Xs[i] shape: (n_samples, n_features_i)
        Notes
        -----
        The fit function performs steps that are common in single-view Classical
        Multidimensional Scaling.

        For each input view :math:'V_i', the Euclidean distance matrix,
        :math:'D_i' is calculated. After this, each matrix is double-centered

        ..math::

            {\textstyle B_i=-\frac{1}{2}J_iD_i^{(2)}J_i}

        where :math:'B_i' represents the i-th double-centered matrix and
        :math:'J_i' is defined as

        ..math::

            {\textstyle J_i=I_i-{\frac {1}{n}}11_i'}

        Here :math:'I_i' represents an identity matrix with the dimensions of
        :math:'J_i' and :math:'11_i' represents a ones matrix with the
        dimensions of :math:'J_i'.

        After this, the common principal components are found between all
        :math:'D_i' matrices
        
        """

        if (self.n_components) > len(Xs[0]):
            self.n_components = len(Xs[0])
            warnings.warn('The number of components you have requested is '
                          + 'greater than the number of samples in the '
                          + 'dataset. ' + str(self.n_components)
                          + ' components were computed instead.')

        if (self.num_iter) <= 0:
            raise ValueError('The number of iterations must be greater than 0')

        if (self.n_components) <= 0:
            raise ValueError('The number of components must be greater than 0 '
                             + 'and less than the number of features')

        Xs = check_Xs(Xs, multiview=True)

        mat = np.ones(shape=(len(Xs), len(Xs[0]), len(Xs[0])))

        # Double centering each view as in single-view MDS
        for i in np.arange(len(Xs)):
            view = euclidean_distances(Xs[i])
            view_squared = np.power(np.array(view), 2)

            J = np.eye(len(view)) - (1/len(view))*np.ones(view.shape)
            B = -(1/2) * np.matmul(np.matmul(J, view_squared), J)
            mat[i] = B

        self.components = self._commonpcs(mat)

    def fit_transform(self, Xs):

        """"
        Embeds data matrix(s) using fitted projection matrices

        Parameters
        ----------

        Xs: list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            The data to embed based on the fit function.

        Returns
        -------
        components: numpy.ndarray
            - components shape: (n_samples, n_components)
            MVMDS components of Xs
        """
        Xs = check_Xs(Xs)
        self.fit(Xs)

        return self.components
