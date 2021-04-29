from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import  TfidfVectorizer
from atosflow.utils import *
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import numpy as np

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)



class Preprocessing:
    def __init__(self):
        pass
    def preprocessing_text_fit(self,path):
        # getting data from a CSV file
        df = pd.read_csv(path) 
        X_train, X_test, y_train, y_test = train_test_split(
        df['document'], df['target'], test_size=.2, stratify=df['target'], random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=.2, stratify=y_train, random_state=42)
        vec = TfidfVectorizer(
                    input='array',
                    norm='l2',
                    max_features=None,
                    sublinear_tf=True,
                    stop_words='english')

        X_train_transformed= tf.sparse.reorder(convert_sparse_matrix_to_sparse_tensor(vec.fit_transform(X_train)))
        X_val_transformed = tf.sparse.reorder(convert_sparse_matrix_to_sparse_tensor(vec.transform(X_val)))
        X_test_transformed= tf.sparse.reorder(convert_sparse_matrix_to_sparse_tensor(vec.transform(X_test)))
        le = preprocessing.LabelEncoder()
        y_train = le.fit_transform(y_train)
        
        y_val = le.transform(y_val)
        y_test = le.transform(y_test)
        self.le = le
        self.vec = vec

        return X_train_transformed,X_test_transformed,X_val_transformed,y_train,y_val,y_test
    
    def preprocess(self,text):
        return self.vec.transform(text)

