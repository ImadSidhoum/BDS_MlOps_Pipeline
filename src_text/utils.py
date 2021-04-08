import numpy as np
import pandas as pd
import pickle
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import mlflow
dataset = load_dataset('allocine')
print("datasets loaded")

# Reviews need to be tokenized
X_train = np.array(dataset["train"]['review'])
X_val = np.array(dataset["validation"]['review'])
X_test = np.array(dataset["test"]['review'])

y_train = dataset["train"]['label']
y_val = dataset["validation"]['label']
y_test = dataset["test"]['label']
class_names = ['Positive','Negative']

# enable autologging
mlflow.sklearn.autolog()

tfidf_clf = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2),
        max_df=0.75
    )),
    ('clf', LogisticRegression(
        C=1300, penalty='l2', 
        n_jobs=-1, verbose=1
    )),
])


print("training")
with mlflow.start_run() as run:
    for i in range(2):
        tfidf_clf.fit(X_train, y_train)

