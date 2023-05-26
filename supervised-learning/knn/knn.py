import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from typing import List


class KNNClassifier:
    def __init__(self, feature_indices: List[int]=[0, 1], n_neighbors: int=15, weights: str='uniform', 
                 test_size: float=0.2, random_state: int=42):
        """
        Initialize the KNNClassifier with specified parameters.

        Parameters
        ----------
        feature_indices : list of int
            The indices of the features to use for training and prediction.
        n_neighbors : int
            The number of neighbors to use for the k-NN algorithm.
        weights : str
            The weight function to use for prediction. Possible values are 'uniform' and 'distance'.
        test_size : float
            The proportion of the dataset to include in the test split.
        random_state : int
            The seed used by the random number generator.
        """
        self.feature_indices = feature_indices
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.test_size = test_size
        self.random_state = random_state

    def load_dataset(self) -> None:
        """
        Load the Iris dataset.
        """
        iris = datasets.load_iris()
        self.X = iris.data[:, self.feature_indices]
        self.y = iris.target

    def split_dataset(self) -> None:
        """
        Split the dataset into training and test sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def train_model(self) -> None:
        """
        Train the k-NN classifier.
        """
        self.clf = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)
        self.clf.fit(self.X_train, self.y_train)

    def evaluate_model(self) -> None:
        """
        Evaluate the model by computing the accuracy on the test set.
        """
        score = self.clf.score(self.X_test, self.y_test)
        print("Model accuracy: ", score)

    def plot_decision_boundary(self) -> None:
        """
        Plot the decision boundary of the k-NN classifier.
        """
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
        
        Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=cmap_bold, edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i)" % (self.n_neighbors))
        plt.show()
