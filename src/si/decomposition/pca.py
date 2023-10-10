import numpy as np

from si.data.dataset import Dataset


class PCA:
    
    def __init__(self, n_components: int):

        """
        PCA used to reduce the dimensions of the dataset.

        Parameters
    ----------
        n_components: int
                    Number of components
        
        Attributes
        ----------
        mean - mean of the samples
        components - the principal components (the unitary matrix of eigenvectors)
        explained_variance - explained variance (diagonal matrix of eigenvalues)
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset: Dataset) -> np.ndarray:
        """
        Estimates the mean, principal components and explained variance.

        Parameters
        ----------
        dataset: Dataset
               A labeled dataset
        
        Returns
        self
        """
        #centering the data
        self.mean = np.mean(dataset.X, axis = 0)
        dataset = dataset.X - self.mean

        #calculate of SVD
        U,S,V = np.linalg.svd(dataset, full_matrices=False) #U: unitary matrix of eigenvectors
                                                            #S: diagonal matrix of eigenvalues
                                                            #V: unitary matrix of right singular vectors
        
        #infer the Principal Components
        self.components = V[:self.n_components]

        #infer the Explained Variance
        n_samples = dataset.shape[0]
        EV = (self.S ** 2)/(n_samples - 1)
        self.explained_variance = EV[:self.n_components]

        return self
    
    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        Transforms dataset by calculating the reduction of X to the principal components.

        Parameters
        ----------
        dataset: Dataset
               A labeled dataset
        
        Returns
        Reduced Dataset
        """
        dataset = dataset.X - self.mean
        
        v_matrix = self.components.T

        reduced_data = np.dot(dataset, v_matrix)

        return Dataset(reduced_data, dataset.y, dataset.features, dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Calculates the mean, the principal components and the explained variance using SVD and 
        calculates the reduced dataset.

        Returns: 
        Dataset object.
        """
        self.fit(dataset)
        return self.transform(dataset)