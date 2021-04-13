import requests
import tensorflow as tf
import json
from datasets import load_dataset
import tensorflow_datasets as tfds

import sys


train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)



textes = test_data[:10].tolist()


data = json.dumps({"signature_name": "serving_default", "text": textes})
headers = {"content-type": "application/json"}
json_response = requests.post(f'http://127.0.0.1:5001/predictTextClassification', data=data, headers=headers)
js = json_response.text 
if js:
    predictions = js
    print(predictions)
else:
    print("not enough text")