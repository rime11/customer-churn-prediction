from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class CustomFeatureProcessor(BaseEstimator, TransformerMixin):
    '''
    Create a custom transformer for the scale_encode_data function by
    inheriting from the two base classes to create a scikit learn compatible transformer, 
    they provide necessary methods to work with the pipeline API
    
    '''
    def __init__(self, processing_func, non_num_cols, is_train = True, scaler=False):
        self.processing_func = processing_func
        self.non_num_cols = non_num_cols
        self.is_train=is_train
        self.scaler = scaler

    def fit(self, X, y=None):
        #used during model training where the scaler learns the mean and standard deviation of each feature.
        #y=None allows the transformer to be used in supervised and unsupervised 
        if self.is_train:
            X_processed, self.scaler = self.processing_func(X, non_num_cols=self.non_num_cols, 
                                                            is_train=self.is_train, scaler = self.scaler)
        return self

    def transform(self, X, y=None):
        #uses existing scaler (self.scaler) for training by setting is_train to false and return processed_X
        X_processed, _ = self.processing_func(X, non_num_cols=self.non_num_cols, 
                                              is_train=False, scaler = self.scaler)
        return X_processed