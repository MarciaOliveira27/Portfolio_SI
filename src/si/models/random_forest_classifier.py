from typing import Literal, Counter

import numpy as np
from si.data.dataset import Dataset
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.metrics.accuracy import accuracy

class RandomForestClassifier:
    """
    Class representing a technique that combines multiple decision trees to 
    improve prediction accuracy and reduce overfitting.
    """
    def __init__(self, n_estimators: int = 100, max_features: int = None, min_sample_split: int = 2,
                max_depth: int = 15, mode: Literal['gini', 'entropy'] = 'gini', 
                seed: int = None) -> None:
        """
        Creates a RandomForestClassifier object.

        Parameters
        ----------
        n_estimators: int
            number of decision trees to use
        max_features: int
            maximum number of features to use per tree 
        min_sample_split: int
            minimum samples allowed in a split
        max_depth: int
            maximum depth of the trees
        mode: Literal['gini', 'entropy']
            the mode to use for calculating the information gain
        seed: int
            the seed to use for the random number generator
        """
        #parameters
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        #estimated parameters
        self.trees = []
        self.training = {}

    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Fits the random forest classifier to a dataset and 
        train the decision trees of the random forest.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.

        Returns
        -------
        RandomForestClassifier
            The fitted model.
        """

        if self.seed is not None:
            np.random.seed(self.seed)

        n_samples, n_features = dataset.shape()
        
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        #create a bootsrap dataset
        for x in range(self.n_estimators):
            #n_samples random samples from the dataset with replacement
            random_samples = np.random.choice(n_samples, n_samples, replace = True)
            #self.max_features random features without replacement
            random_features = np.random.choice(n_features, self.max_features, replace = False)

            #create and train a decision tree with the bootstrap dataset
            bootstrap_dataset = Dataset(dataset.X[random_samples][:,random_features], dataset.y[random_samples])

            #tuple containing the features used and the trained tree
            trained_tree = DecisionTreeClassifier(min_sample_split = self.min_sample_split, 
                                                  max_depth = self.max_depth, mode = self.mode)

            trained_tree.fit(bootstrap_dataset)

            self.trees.append((random_features, trained_tree))
       
        return self
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Makes predictions about the labels using the ensemble models.

        Parameters
        ----------
        dataset: Dataset
            The dataset to make predictions for.

        Returns
        -------
        np.ndarray
            The predicted labels.
        """
        n_samples = dataset.shape()[0]

        #predictions for each tree using the respective set of features
        all_final_predictions = np.zeros((self.n_estimators, n_samples), dtype=object)

        for tree_idx, (features_idx, trained_tree) in enumerate(self.trees):
            selected_data = Dataset(dataset.X[:, features_idx], dataset.y)
            predictions_tree = trained_tree.predict(selected_data)
            all_final_predictions[tree_idx, :] = predictions_tree
        
        return all_final_predictions
    
    def score(self, dataset: Dataset) -> float:
        """
        Calculates the accuracy of the model on a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to calculate the accuracy on.

        Returns
        -------
        float
            The accuracy of the model on the dataset.
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)