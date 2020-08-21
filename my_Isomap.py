import numpy as np
from sklearn.neighbors import NearestNeighbors as KNN  # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import pairwise_distances
from numpy import linalg as LA
from sklearn.utils.graph_shortest_path import graph_shortest_path #--> for geodesic distance --> https://scikit-learn.org/stable/modules/generated/sklearn.utils.graph_shortest_path.graph_shortest_path.html

# ----- python fast kernel matrix:
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
# https://stackoverflow.com/questions/7391779/fast-kernel-matrix-computation-python
# https://stats.stackexchange.com/questions/15798/how-to-calculate-a-gaussian-kernel-effectively-in-numpy
# https://stackoverflow.com/questions/36324561/fast-way-to-calculate-kernel-matrix-python?rq=1

# ----- python fast scatter matrix:
# https://stackoverflow.com/questions/31145918/fast-weighted-scatter-matrix-calculation

# ----- Isomap in python:
# https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/manifold/isomap.py#L15
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
# http://benalexkeen.com/isomap-for-dimensionality-reduction-in-python/ ---> also includes dataset of faces of statue

class My_Isomap:

    def __init__(self, n_components=None, n_neighbors=5, n_jobs=-1):
        self.n_components = n_components
        self.X = None
        self.Delta_squareRoot = None
        self.V = None
        self.geodesic_dist_matrix = None
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs # number of parallel jobs --> -1 means all processors --> https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors_graph
        self.kernel_training = None


    def fit_transform(self, X):
        # X: rows are features and columns are samples
        self.fit(X)
        X_transformed = self.transform(X)
        return X_transformed

    def fit(self, X):
        # X: rows are features and columns are samples
        self.X = X
        n_samples = self.X.shape[1]
        self.geodesic_dist_matrix = self.find_geodesic_distance_matrix(X=self.X)
        H = np.eye(n_samples) - ((1 / n_samples) * np.ones((n_samples, n_samples)))
        self.kernel_training = -0.5 * (H.dot(self.geodesic_dist_matrix).dot(H))
        # V, delta, Vh = LA.svd(self.kernel_training, full_matrices=True)
        eig_val, eig_vec = np.linalg.eigh(self.kernel_training)
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        delta = eig_val[idx]
        V = eig_vec[:, idx]
        if self.n_components != None:
            V = V[:,:self.n_components]
            delta = delta[:self.n_components]
        delta = np.asarray(delta)
        delta_squareRoot = delta**0.5
        self.Delta_squareRoot = np.diag(delta_squareRoot)
        self.V = V

    def transform(self, X):
        # X: rows are features and columns are samples
        X_transformed = (self.Delta_squareRoot).dot(self.V.T)
        return X_transformed

    def find_geodesic_distance_matrix(self, X):
        # ----- find k-nearest neighbor graph (distance matrix):
        if self.n_neighbors == None:
            n_samples = X.shape[1]
            self.n_neighbors = n_samples
        knn = KNN(n_neighbors=self.n_neighbors+1, algorithm='kd_tree', n_jobs=self.n_jobs)  #+1 because the point itself is also counted
        knn.fit(X=X.T)
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors_graph
        # the following function gives n_samples*n_samples matrix, and puts 0 for diagonal and also where points are not connected directly in KNN graph
        # if K=n_samples, only diagonal is zero.
        Euclidean_distance_matrix = knn.kneighbors_graph(X=X.T, n_neighbors=self.n_neighbors, mode='distance') #--> gives Euclidean distances
        #Euclidean_distance_matrix = Euclidean_distance_matrix.toarray()
        # ----- find geodesic distance graph:
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.graph_shortest_path.graph_shortest_path.html
        geodesic_dist_matrix = graph_shortest_path(dist_matrix=Euclidean_distance_matrix, method="auto", directed=False)
        return geodesic_dist_matrix

    def transform_outOfSample(self, X_test):
        # X_test: rows are features and columns are samples
        n_test_samples = X_test.shape[1]
        n_training_samples = self.X.shape[1]
        #--> the following is not completely correct as only training points should be used for intermediate points in geodesic distance calculation, but I approximate it:
        geodesic_distance_matrix_TrainTest = self.find_geodesic_distance_matrix(X=np.column_stack((self.X, X_test)))
        geodesic_distance_matrix_TrainWithRespectToTest = geodesic_distance_matrix_TrainTest[:n_training_samples, n_training_samples:]
        n_components = self.V.shape[1]
        X_test_transformed = np.zeros((n_components, n_test_samples))
        for test_sample_index in range(n_test_samples):
            dist = geodesic_distance_matrix_TrainWithRespectToTest[:, test_sample_index]
            dist_average = np.mean(self.geodesic_dist_matrix)
            for dimension_index in range(n_components):
                v = self.V[:, dimension_index] #--> n-dimensional vector
                k = dist_average - dist #--> n-dimensional vector
                eig_values_squareRoot = np.diag(self.Delta_squareRoot)
                delta_ = eig_values_squareRoot ** 2
                X_test_transformed[dimension_index, test_sample_index] = (1/(2 * delta_[dimension_index]**0.5)) * (v.T@k)
        return X_test_transformed
        