import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier:
    """
    Stacking classifier harnesses an ensemble of models to generate predictions.
    These predictions are subsequently employed to train another model (final model).
    The final model can then be used to predict the output variable (Y).

    Parameters
    ----------
    models : array-like, shape = [n_models]
        Initial set of models.
    final_model:
        The model to make the final predictions.

    Attributes
    ----------
    """
    def __init__(self, models, final_model):
        """
        Initialize the stacking classifier.

        Parameters
        ----------
        models: array-like, shape = [n_models]
            Initial set of models.
        final_model:
            The model to make the final predictions.

        """
        #parameters
        self.models = models
        self.final_model = final_model

    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the models according to the given training data.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        self : StackingClassifier
            The fitted model.
        """

        for model in self.models:
            model.fit(dataset)

        predictions = []

        for model in self.models:
            predictions.append(model.predict(dataset))
        
        predictions = np.array(predictions).T
        #train the final model with the predictions of the initial set of models
        self.final_model.fit(Dataset(dataset.X, predictions))

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for samples in X using the ensemble models.

        Parameters
        ----------
        dataset : Dataset
            The test data.

        Returns
        -------
        y : array-like, shape = [n_samples]
            The predicted class labels.
        """
        
        #get predictions from the initial set of models
        initial_predictions = []
        
        for model in self.models:
            initial_predictions.append(model.predict(dataset))
        
        initial_predictions = np.array(initial_predictions).T
        
        #get the final predictions using the final model and the predictions of the initial set of models
        final_predictions = self.final_model.predict(Dataset(dataset.X, initial_predictions))
        
        return final_predictions
    
    def score(self, dataset: Dataset) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        dataset : Dataset
            The test data.

        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, self.predict(dataset))
