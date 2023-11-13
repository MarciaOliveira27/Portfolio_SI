from typing import Dict, Tuple, Callable, Any

import numpy as np
from si.data.dataset import Dataset
from si.model_selection.cross_validation import k_fold_cross_validation

def randomized_search_cv(model, dataset: Dataset, hyperparameter_grid: Dict[str, Tuple], scoring: Callable = None, 
                         cv: int = 5, n_iter: int = 10) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation on the given model and dataset.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    hyperparameter_grid:
        Dictionary with the hyperparameter name and search values.
    scoring: Callable
        The scoring function to use. If None, the model's score method will be used.
    cv: int
        The number of cross-validation folds.
    n_iter: int
        Number of hyperparameter random combinations to test

    Returns
    -------
    scores: Dict[str, Any]
        Results of the randomized search cross validation, includes the hyperparameters, 
        scores, best hyperparameters and best score.
    """

    #check if the hyperparameters are valid
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")
    
    results = {'hyperparameters': [] , 'scores': []}
    
    for combination in range(n_iter):

        model_hyperparameters = {}

        for param, values in hyperparameter_grid.items():
            model_hyperparameters[param] = np.random.choice(values)

        for parameter, value in model_hyperparameters.items():
            setattr(model, parameter, value)

        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        results['hyperparameters'].append(model_hyperparameters)
        results['scores'].append(np.mean(score))

    results['best_hyperparameters'] = results['hyperparameters'][np.argmax(results['scores'])]
    results['best_score'] = np.max(results['scores'])
    
    return results


        


