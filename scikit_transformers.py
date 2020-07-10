import numpy as np
import pandas as pd
from re import search

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer


class FeatureSelector(BaseEstimator, TransformerMixin):
    #Class Constructor 
    def __init__(self, feature_names):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit(self, X, y = None):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform(self, X, y = None):
        return X[self._feature_names] 


#converts certain features to binary 
class MetaTextTransformer(TransformerMixin):
        
    #Return self nothing else to do here
    def fit( self, X, y = None  ):
        return self

    #Helper function to extract AP courses from text
    def get_ap(self, obj):
        if search(r"^.*?(advanced placement)", obj):
            return "yes"
        else:
            return "No"

    def get_ell(self, obj):
        if search(r"^.*?(language learners)", obj):
            return "yes"
        else:
            return "No"
    
    #Helper function that converts values to Binary depending on boolean input 
    def create_binary(self, obj):
        if obj == 0:
            return 'No'
        else:
            return 'Yes'
    
    #Transformer method we wrote for this transformer 
    def transform(self, X , y = None ):
       #using the helper functions written above 
       ap = X.apply(self.get_ap)

       ell = X.apply(self.get_ell)
       
       combine_feats = pd.concat([ap, ell], axis=1)
       #returns numpy array
       return combine_feats.values