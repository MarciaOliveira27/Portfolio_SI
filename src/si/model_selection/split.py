from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    In a stratified manner, split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train_data: Dataset
        The training dataset
    test_data: Dataset
        The testing dataset
    """
    
    unique_labels, counts_labels = np.unique(dataset.y, return_counts=True)

    train_idx = []
    test_idx = []

    np.random.seed(random_state)

    for labels, counts in zip(unique_labels, counts_labels):
        #get number of samples in the test set
        number_test = int(counts * test_size)                  
        #select indices for the current class
        idx_labels = np.where(dataset.y == labels)[0]          
        #perform a random shuffle
        np.random.shuffle(idx_labels)                          
        #add indices to the list test_idx
        test_idx.extend(idx_labels[:number_test])              
        #add indices to the list train_idx
        train_idx.extend(idx_labels[number_test:])             
    
    #create training and testing datasets
    train_data = Dataset(dataset.X[train_idx], dataset.y[train_idx], features=dataset.features, label=dataset.label)
    test_data = Dataset(dataset.X[test_idx], dataset.y[test_idx], features=dataset.features, label=dataset.label)
    
    return train_data, test_data