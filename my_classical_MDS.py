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

    def transform_outOfSample(self, X_test):
        # X_test: rows are features and columns are samples
        # kernel_X_X = pairwise_kernels(X=self.X.T, Y=self.X.T, metric=self.kernel)
        kernel_X_Xtest = pairwise_kernels(X=self.X.T, Y=X_test.T, metric=self.kernel)
        kernel_outOfSample_centered = self.center_kernel_of_outOfSample(kernel_of_outOfSample=kernel_X_Xtest, matrix_or_vector="matrix")
        n_components = self.V.shape[1]
        n_test_samples = X_test.shape[1]
        X_test_transformed = np.zeros((n_components, n_test_samples))
        for test_sample_index in range(n_test_samples):
            for dimension_index in range(n_components):
                v = self.V[:, dimension_index] #--> n-dimensional vector
                k = kernel_outOfSample_centered[:, test_sample_index] #--> n-dimensional vector
                eig_values_squareRoot = np.diag(self.Delta_squareRoot)
                delta_ = eig_values_squareRoot ** 2
                X_test_transformed[dimension_index, test_sample_index] = (1/(delta_[dimension_index]**0.5)) * (v.T@k)
        return X_test_transformed

    def center_kernel_of_outOfSample(self, kernel_of_outOfSample, matrix_or_vector="matrix"):
        n_training_samples = self.X.shape[1]
        kernel_X_X_training = pairwise_kernels(X=self.X.T, Y=self.X.T, metric=self.kernel)
        if matrix_or_vector == "matrix":
            n_outOfSample_samples = kernel_of_outOfSample.shape[1]
            kernel_of_outOfSample_centered = kernel_of_outOfSample - (1 / n_training_samples) * np.ones((n_training_samples, n_training_samples)).dot(kernel_of_outOfSample) \
                                             - (1 / n_training_samples) * kernel_X_X_training.dot(np.ones((n_training_samples, n_outOfSample_samples))) \
                                             + (1 / n_training_samples**2) * np.ones((n_training_samples, n_training_samples)).dot(kernel_X_X_training).dot(np.ones((n_training_samples, n_outOfSample_samples)))
        elif matrix_or_vector == "vector":
            kernel_of_outOfSample = kernel_of_outOfSample.reshape((-1, 1))
            kernel_of_outOfSample_centered = kernel_of_outOfSample - (1 / n_training_samples) * np.ones((n_training_samples, n_training_samples)).dot(kernel_of_outOfSample) \
                                             - (1 / n_training_samples) * kernel_X_X_training.dot(np.ones((n_training_samples, 1))) \
                                             + (1 / n_training_samples**2) * np.ones((n_training_samples, n_training_samples)).dot(kernel_X_X_training).dot(np.ones((n_training_samples, 1)))
        return kernel_of_outOfSample_centered

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
        