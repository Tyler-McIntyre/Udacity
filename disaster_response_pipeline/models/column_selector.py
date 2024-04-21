import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.key]
        elif isinstance(X, np.ndarray):
            if len(X.shape) == 1:
                return X
            else:
                return X[:, self.key]
        else:
            raise ValueError("Unsupported input type. Expected DataFrame or NumPy array.")