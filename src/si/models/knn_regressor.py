from typing import Callable, Union

import numpy as np

from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from si.metrics.rmse import rmse


class KNNRegressor:
    """
    The k-Nearest Neighbors Regressor is suitable for regression problems. Therefore, it estimates 
    the average value of the k most similar examples.

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use that calculates the distance between a sample 
        and the samples in the training dataset

    Attributes
    ----------
    dataset: np.ndarray
        The training data
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initialize the KNN regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        #parameters/atributes
        self.k = k
        self.distance = distance
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self
    
    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the closest label of the given sample

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of

        Returns
        -------
        label: str or int
            The closest label
        """
        #compute the distance between each sample and various samples in the training dataset
        distances = self.distance(sample, self.dataset.X)

        #get the indexes of the k most similar examples (shortest distance)
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        #use the previous indexes to retrieve the corresponding values in Y
        k_nearest_neighbors_values = self.dataset.y[k_nearest_neighbors]

        #calculate the average of the values obtained in previous step 
        mean_values = np.mean(k_nearest_neighbors_values)
        return mean_values

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It estimates the label value for a sample based on the k most similar examples

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model (an array of predicted values for the 
            testing dataset (y_pred))
        """
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        It calculates the error (rmse) between the estimated values and the real ones 

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        rmse: float
            Rmse error - error between predictions and actual values
        """
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)


if __name__ == '__main__':
    #import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    #load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    #initialize the KNN Regressor
    knn = KNNRegressor(k=3)

    #fit the model to the train dataset
    knn.fit(dataset_train)

    #evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The rmse of the model is: {score}')