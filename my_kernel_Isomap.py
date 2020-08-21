import numpy as np
from sklearn.neighbors import NearestNeighbors as KNN  # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import pairwise_distances
from numpy import linalg as LA
from sklearn.utils.graph_shortest_path import graph_shortest_path #--> for geodesic distance --> https://scikit-learn.org/stable/modules/generated/sklearn.utils.graph_shortest_path.graph_shortest_path.html
import sys


class My_kernel_Isomap:

    def __init__(self, n_components=None, n_neighbors=5, n_jobs=-1):
        self.n_components = n_components
        self.X = None
        self.Delta_squareRoot = None
        self.V = None
        self.geodesic_dist_matrix = None
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs # number of parallel jobs --> -1 means all processors --> https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors_graph
        self.kernel_prime = None

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
        kernel_D = -0.5 * (H.dot(self.geodesic_dist_matrix).dot(H))
        kernel_D_squared = -0.5 * (H.dot(self.geodesic_dist_matrix@self.geodesic_dist_matrix).dot(H))
        criterion_matrix1 = np.column_stack(( np.zeros((n_samples,n_samples)), 2*kernel_D_squared ))
        criterion_matrix2 = np.column_stack(( -1*np.ones((n_samples,n_samples)), -4*kernel_D ))
        criterion_matrix = np.vstack(( criterion_matrix1, criterion_matrix2 ))
        eig_val, eig_vec = np.linalg.eigh(criterion_matrix)
        c_star = np.max(eig_val)
        self.kernel_prime = kernel_D_squared + (2*c_star*kernel_D) + (0.5*(c_star**2)*H)
        eig_val, eig_vec = np.linalg.eigh(self.kernel_prime)
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





    # def transform_outOfSample(self, X_test):
    #     # X_test: rows are features and columns are samples
    #     n_test_samples = X_test.shape[1]
    #     n_training_samples = self.X.shape[1]
    #     #--> the following is not completely correct as only training points should be used for intermediate points in geodesic distance calculation, but I approximate it:
    #     geodesic_distance_matrix_TrainTest = self.find_geodesic_distance_matrix(X=np.column_stack((self.X, X_test)))
    #     n_samples_train_and_test = n_training_samples + n_test_samples
    #     H = np.eye(n_samples_train_and_test) - ((1 / n_samples_train_and_test) * np.ones((n_samples_train_and_test, n_samples_train_and_test)))
    #     kernel_training_test = -0.5 * (H.dot(geodesic_distance_matrix_TrainTest).dot(H))
    #     kernel_training_test = kernel_training_test[:n_training_samples, n_training_samples:]  #--> make size (n_training_samples, n_test_samples)
    #     kernel_outOfSample_centered = self.center_kernel_of_outOfSample(kernel_of_outOfSample=kernel_training_test, kernel_training=self.kernel_prime, matrix_or_vector="matrix")
    #     self.Delta_squareRoot[np.isnan(self.Delta_squareRoot)] = 0
    #     # X_test_transformed = np.linalg.inv(self.Delta_squareRoot) @ self.V.T @ kernel_outOfSample_centered
    #     X_test_transformed = self.safe_inverse_matrix(self.Delta_squareRoot) @ self.V.T @ kernel_outOfSample_centered
    #     return X_test_transformed

    # def center_kernel_of_outOfSample(self, kernel_of_outOfSample, kernel_training, matrix_or_vector="matrix"):
    #     n_training_samples = self.X.shape[1]
    #     # kernel_X_X_training = pairwise_kernels(X=self.X.T, Y=self.X.T, metric=self.kernel)
    #     kernel_X_X_training = kernel_training
    #     if matrix_or_vector == "matrix":
    #         n_outOfSample_samples = kernel_of_outOfSample.shape[1]
    #         kernel_of_outOfSample_centered = kernel_of_outOfSample - (1 / n_training_samples) * np.ones((n_training_samples, n_training_samples)).dot(kernel_of_outOfSample) \
    #                                          - (1 / n_training_samples) * kernel_X_X_training.dot(np.ones((n_training_samples, n_outOfSample_samples))) \
    #                                          + (1 / n_training_samples**2) * np.ones((n_training_samples, n_training_samples)).dot(kernel_X_X_training).dot(np.ones((n_training_samples, n_outOfSample_samples)))
    #     elif matrix_or_vector == "vector":
    #         kernel_of_outOfSample = kernel_of_outOfSample.reshape((-1, 1))
    #         kernel_of_outOfSample_centered = kernel_of_outOfSample - (1 / n_training_samples) * np.ones((n_training_samples, n_training_samples)).dot(kernel_of_outOfSample) \
    #                                          - (1 / n_training_samples) * kernel_X_X_training.dot(np.ones((n_training_samples, 1))) \
    #                                          + (1 / n_training_samples**2) * np.ones((n_training_samples, n_training_samples)).dot(kernel_X_X_training).dot(np.ones((n_training_samples, 1)))
    #     return kernel_of_outOfSample_centered

    # def safe_inverse_matrix(self, matrix_):
    #     # https://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
    #     if np.linalg.cond(matrix_) < 1/sys.float_info.epsilon:
    #         inv_matrix = np.linalg.inv(matrix_)
    #     else:
    #         matrix_ += np.eye(matrix_.shape[0]) * 1e-5
    #         inv_matrix = np.linalg.inv(matrix_)
    #     return inv_matrix
        