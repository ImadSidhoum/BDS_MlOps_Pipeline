import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import mlflow
import pickle,sys
from datasets import load_dataset
sys.path.insert(0, '../../atosflow')
from utils import *

# Split the training set into 60% and 40% to end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.

train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

print('data loaded')

# enable autologging
mlflow.tensorflow.autolog()

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)


model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("training")
with mlflow.start_run() as run:
    history = model.fit(train_data.shuffle(10000).batch(512),
                        epochs=1,
                        validation_data=validation_data.batch(512),
                        verbose=1)

    results = model.evaluate(test_data.shuffle(10000).batch(512))
    mlflow.log_metric("test_loss", results[0])
    mlflow.log_metric("test_accuracy", results[1])
print("test loss, test acc:", results)

compare(run.info.run_uuid,name='text')  
print('fin')
