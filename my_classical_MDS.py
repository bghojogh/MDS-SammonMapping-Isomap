import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import pairwise_distances
from numpy import linalg as LA

# ----- python fast kernel matrix:
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
# https://stackoverflow.com/questions/7391779/fast-kernel-matrix-computation-python
# https://stats.stackexchange.com/questions/15798/how-to-calculate-a-gaussian-kernel-effectively-in-numpy
# https://stackoverflow.com/questions/36324561/fast-way-to-calculate-kernel-matrix-python?rq=1

# ----- python fast scatter matrix:
# https://stackoverflow.com/questions/31145918/fast-weighted-scatter-matrix-calculation


class My_classical_MDS:

    def __init__(self, n_components=None, kernel='linear'):
        self.n_components = n_components
        self.X = None
        self.Delta_squareRoot = None
        self.V = None
        self.kernel = kernel

    def fit_transform(self, X):
        # X: rows are features and columns are samples
        self.fit(X)
        X_transformed = self.transform(X)
        return X_transformed

    def fit(self, X):
        # X: rows are features and columns are samples
        self.X = X
        kernel_X_X = pairwise_kernels(X=X.T, Y=X.T, metric=self.kernel)
        kernel_X_X = self.center_the_matrix(the_matrix=kernel_X_X, mode="double_center")
        # V, delta, Vh = np.linalg.svd(kernel_X_X, full_matrices=True)
        eig_val, eig_vec = np.linalg.eigh(kernel_X_X)
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

    def center_the_matrix(self, the_matrix, mode="double_center"):
        n_rows = the_matrix.shape[0]
        n_cols = the_matrix.shape[1]
        vector_one_left = np.ones((n_rows,1))
        vector_one_right = np.ones((n_cols, 1))
        H_left = np.eye(n_rows) - ((1/n_rows) * vector_one_left.dot(vector_one_left.T))
        H_right = np.eye(n_cols) - ((1 / n_cols) * vector_one_right.dot(vector_one_right.T))
        if mode == "double_center":
            the_matrix = H_left.dot(the_matrix).dot(H_right)
        elif mode == "remove_mean_of_rows_from_rows":
            the_matrix = H_left.dot(the_matrix)
        elif mode == "remove_mean_of_columns_from_columns":
            the_matrix = the_matrix.dot(H_right)
        return the_matrix