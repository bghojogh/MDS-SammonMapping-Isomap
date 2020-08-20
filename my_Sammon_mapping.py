from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.neighbors import NearestNeighbors as KNN  # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.decomposition import PCA


import warnings
warnings.filterwarnings("ignore")

class My_Sammon_mapping:

    def __init__(self, X, n_components, n_neighbors=None, 
                 max_iterations=100, learning_rate=0.1, init_type="PCA"):
        self.embedding_dimensionality = n_components
        self.n_neighbors = n_neighbors
        self.n_neighbors = n_neighbors
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.init_type = init_type  #--> PCA, random

    def fit_transform(self, X):
        # X: samples are put column-wise in matrix
        self.X = X
        self.n_dimensions = X.shape[0]
        self.n_samples = X.shape[1]
        if self.n_neighbors is None:
            self.n_neighbors = self.n_samples - 1
        X_transformed = self.Quasi_Newton_optimization(X=self.X, max_iterations=self.max_iterations)
        return X_transformed

    def Quasi_Newton_optimization(self, X, max_iterations=100):
        # X: column-wise samples
        iteration_start = 0
        if self.init_type == "random":
            X_low_dim = np.random.rand(self.embedding_dimensionality, self.n_samples)  # --> rand in [0,1)
        elif self.init_type == "PCA":
            pca = PCA(n_components=self.embedding_dimensionality)
            X_low_dim = (pca.fit_transform(X.T)).T
        KNN_distance_matrix_initial, neighbors_indices = self.find_KNN_distance_matrix(X=X, n_neighbors=self.n_neighbors)
        normalization_factor = sum(sum(KNN_distance_matrix_initial))
        for iteration_index in range(iteration_start, max_iterations):
            print("Performing quasi Newton, iteration " + str(iteration_index))
            All_NN_distance_matrix, _ = self.find_KNN_distance_matrix(X=X_low_dim, n_neighbors=self.n_samples-1)
            for sample_index in range(self.n_samples):
                for dimension_index in range(self.embedding_dimensionality):
                    # --- calculate gradient and second derivative of gradient (Hessian):
                    gradient_term = 0.0
                    Hessian_term = 0.0
                    for neighbor_index in range(self.n_neighbors):
                        neighbor_index_in_dataset = neighbors_indices[sample_index, neighbor_index]
                        d = All_NN_distance_matrix[sample_index, neighbor_index_in_dataset]
                        d_initial = KNN_distance_matrix_initial[sample_index, neighbor_index_in_dataset]
                        gradient_term += ((d - d_initial) / (d * d_initial)) * (X_low_dim[dimension_index, sample_index] - X_low_dim[dimension_index, neighbor_index_in_dataset])
                        Hessian_term += ((d - d_initial) / (d * d_initial)) - ((X_low_dim[dimension_index, sample_index] - X_low_dim[dimension_index, neighbor_index_in_dataset])**2 / d**3)
                    gradient_term *= (1 / normalization_factor)
                    Hessian_term *= (1 / normalization_factor)
                    gradient_ = gradient_term
                    Hessian_ = Hessian_term
                    # --- update solution:
                    X_low_dim[dimension_index, sample_index] = X_low_dim[dimension_index, sample_index] - (self.learning_rate * abs(1/Hessian_) * gradient_)
            # calculate the objective function:
            objective_function_distance_part = 0.0
            for sample_index in range(self.n_samples):
                temp_ = 0.0
                for neighbor_index in range(self.n_neighbors):
                    neighbor_index_in_dataset = neighbors_indices[sample_index, neighbor_index]
                    d = All_NN_distance_matrix[sample_index, neighbor_index_in_dataset]
                    d_initial = KNN_distance_matrix_initial[sample_index, neighbor_index_in_dataset]
                    temp_ += (d - d_initial)**2 / d_initial
                objective_function_distance_part += (1 / normalization_factor) * temp_
            objective_function = 0.5 * objective_function_distance_part
            print("iteration " + str(iteration_index) + ": objective cost = " + str(objective_function))
        return X_low_dim

    def find_KNN_distance_matrix(self, X, n_neighbors):
        # X: column-wise samples
        # returns KNN_distance_matrix: row-wise --> shape: (n_samples, n_samples) where zero for not neighbors
        # returns neighbors_indices: row-wise --> shape: (n_samples, n_neighbors)
        knn = KNN(n_neighbors=n_neighbors+1, algorithm='kd_tree', n_jobs=-1)  #+1 because the point itself is also counted
        knn.fit(X=X.T)
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors_graph
        # the following function gives n_samples*n_samples matrix, and puts 0 for diagonal and also where points are not connected directly in KNN graph
        # if K=n_samples, only diagonal is zero.
        Euclidean_distance_matrix = knn.kneighbors_graph(X=X.T, n_neighbors=n_neighbors+1, mode='distance') #--> gives Euclidean distances
        KNN_distance_matrix = Euclidean_distance_matrix.toarray()
        neighbors_indices = np.zeros((KNN_distance_matrix.shape[0], n_neighbors))
        for sample_index in range(KNN_distance_matrix.shape[0]):
            neighbors_indices[sample_index, :] = np.ravel(np.asarray(np.where(KNN_distance_matrix[sample_index, :] != 0)))
        neighbors_indices = neighbors_indices.astype(int)
        return KNN_distance_matrix, neighbors_indices

    def remove_outliers(self, data_, color_meshgrid):
        # data_: column-wise samples
        data_outliers_removed = data_.copy()
        color_meshgrid_outliers_removed = color_meshgrid.copy()
        for dimension_index in range(data_.shape[0]):
            data_dimension = data_[dimension_index, :].ravel()
            # Set upper and lower limit to 3 standard deviation
            data_dimension_std = np.std(data_dimension)
            data_dimension_mean = np.mean(data_dimension)
            anomaly_cut_off = data_dimension_std * 3
            lower_limit = data_dimension_mean - anomaly_cut_off
            upper_limit = data_dimension_mean + anomaly_cut_off
            samples_to_keep = []
            for sample_index in range(data_outliers_removed.shape[1]):
                sample_ = data_outliers_removed[:, sample_index]
                if sample_[dimension_index] > upper_limit or sample_[dimension_index] < lower_limit:
                    samples_to_keep.append(False)
                else:
                    samples_to_keep.append(True)
            data_outliers_removed = data_outliers_removed.compress(samples_to_keep, axis=1)
            color_meshgrid_outliers_removed = color_meshgrid_outliers_removed.compress(samples_to_keep)
        return data_outliers_removed, color_meshgrid_outliers_removed

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable

    def save_np_array_to_txt(self, variable, name_of_variable, path_to_save='./'):
        if type(variable) is list:
            variable = np.asarray(variable)
        # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(file_address, 'w') as f:
            f.write(np.array2string(variable, separator=', '))