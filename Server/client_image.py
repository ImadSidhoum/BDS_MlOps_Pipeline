import requests
import tensorflow as tf
import json


import sys


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

images = x_test[:10].tolist()


data = json.dumps({"signature_name": "serving_default", "images": images})
headers = {"content-type": "application/json"}
json_response = requests.post(f'http://127.0.0.1:5001/predictImageClassification', data=data, headers=headers)
js = json_response.text 
if js:
    predictions = js
    print(predictions)
else:
    print("not enough images")