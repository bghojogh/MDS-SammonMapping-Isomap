from my_classical_MDS import My_classical_MDS
from my_Sammon_mapping import My_Sammon_mapping
from Sammon_mapping import sammon
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def main():
    dataset = "MNIST"
    method = "Sammon_mapping" #--> MDS, PCA, my_Sammon_mapping, Sammon_mapping
    X_train, y_train, X_test, y_test, class_names = read_dataset(dataset=dataset)
    if method == "MDS":
        my_classical_MDS = My_classical_MDS()
        X_train_embedded = my_classical_MDS.fit_transform(X=X_train)
    elif method == "PCA":
        pca = PCA(n_components=2)
        X_train_embedded = pca.fit_transform(X=X_train.T)
        X_train_embedded = X_train_embedded.T
    elif method == "my_Sammon_mapping":
        my_Sammon_mapping = My_Sammon_mapping(X=X_train, n_components=2, n_neighbors=None, 
                                            max_iterations=100, learning_rate=0.1, init_type="PCA")
        X_train_embedded = my_Sammon_mapping.fit_transform(X=X_train)
    elif method == "Sammon_mapping":
        [X_train_embedded, E] = sammon(x=X_train.T, n=2, maxiter=10)
        X_train_embedded = X_train_embedded.T    
    color_map = plt.cm.jet  #--> hsv, brg (good for S curve), rgb, jet, gist_ncar (good for one blob), tab10, Set1, rainbow, Spectral #--> https://matplotlib.org/3.2.1/tutorials/colors/colormaps.html
    plt.scatter(X_train_embedded[0, :], X_train_embedded[1, :], c=y_train, cmap=color_map, edgecolors='k')
    classes = class_names
    n_classes = len(classes)
    cbar = plt.colorbar(boundaries=np.arange(n_classes+1)-0.5)
    cbar.set_ticks(np.arange(n_classes))
    cbar.set_ticklabels(classes)
    plt.show()

def read_dataset(dataset):
    if dataset == "MNIST":
        subset_of_MNIST = True
        pick_subset_of_MNIST_again = True
        MNIST_subset_cardinality_training = 200
        MNIST_subset_cardinality_testing = 100
        path_dataset = "./datasets/MNIST/"
        file = open(path_dataset+'X_train.pckl','rb')
        X_train = pickle.load(file); file.close()
        file = open(path_dataset+'y_train.pckl','rb')
        y_train = pickle.load(file); file.close()
        file = open(path_dataset+'X_test.pckl','rb')
        X_test = pickle.load(file); file.close()
        file = open(path_dataset+'y_test.pckl','rb')
        y_test = pickle.load(file); file.close()
        if subset_of_MNIST:
            if pick_subset_of_MNIST_again:
                dimension_of_data = 28 * 28
                X_train_picked = np.empty((0, dimension_of_data))
                y_train_picked = np.empty((0, 1))
                for label_index in range(10):
                    X_class = X_train[y_train == label_index, :]
                    X_class_picked = X_class[0:MNIST_subset_cardinality_training, :]
                    X_train_picked = np.vstack((X_train_picked, X_class_picked))
                    y_class = y_train[y_train == label_index]
                    y_class_picked = y_class[0:MNIST_subset_cardinality_training].reshape((-1, 1))
                    y_train_picked = np.vstack((y_train_picked, y_class_picked))
                y_train_picked = y_train_picked.ravel()
                X_test_picked = np.empty((0, dimension_of_data))
                y_test_picked = np.empty((0, 1))
                for label_index in range(10):
                    X_class = X_test[y_test == label_index, :]
                    X_class_picked = X_class[0:MNIST_subset_cardinality_testing, :]
                    X_test_picked = np.vstack((X_test_picked, X_class_picked))
                    y_class = y_test[y_test == label_index]
                    y_class_picked = y_class[0:MNIST_subset_cardinality_testing].reshape((-1, 1))
                    y_test_picked = np.vstack((y_test_picked, y_class_picked))
                y_test_picked = y_test_picked.ravel()
                # X_train_picked = X_train[0:MNIST_subset_cardinality_training, :]
                # X_test_picked = X_test[0:MNIST_subset_cardinality_testing, :]
                # y_train_picked = y_train[0:MNIST_subset_cardinality_training]
                # y_test_picked = y_test[0:MNIST_subset_cardinality_testing]
                save_variable(X_train_picked, 'X_train_picked', path_to_save=path_dataset)
                save_variable(X_test_picked, 'X_test_picked', path_to_save=path_dataset)
                save_variable(y_train_picked, 'y_train_picked', path_to_save=path_dataset)
                save_variable(y_test_picked, 'y_test_picked', path_to_save=path_dataset)
            else:
                file = open(path_dataset+'X_train_picked.pckl','rb')
                X_train_picked = pickle.load(file); file.close()
                file = open(path_dataset+'X_test_picked.pckl','rb')
                X_test_picked = pickle.load(file); file.close()
                file = open(path_dataset+'y_train_picked.pckl','rb')
                y_train_picked = pickle.load(file); file.close()
                file = open(path_dataset+'y_test_picked.pckl','rb')
                y_test_picked = pickle.load(file); file.close()
            X_train = X_train_picked
            X_test = X_test_picked
            y_train = y_train_picked
            y_test = y_test_picked
        X_train = X_train.T / 255
        X_test = X_test.T / 255
        class_names = [str(i) for i in range(10)]
    return X_train, y_train, X_test, y_test, class_names

def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def load_variable(name_of_variable, path='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable

if __name__ == "__main__":
    main()