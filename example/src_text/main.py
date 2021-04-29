import os
import numpy as np
import tensorflow as tf
import mlflow
import pickle,sys
from tools import *
sys.path.append('../../atosflow')
from utils import *

p = Preprocessing()

X_train_transformed,X_test_transformed,X_val_transformed,y_train,y_val,y_test = p.preprocessing_text_fit('./data.csv')

filename = 'preprocessing.pkl'
outfile = open(filename,'wb')
pickle.dump(p,outfile) 
outfile.close()
'''
infile = open(filename,'rb')
new = pickle.load(infile)
infile.close()
'''
# enable autologging
mlflow.tensorflow.autolog()
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("training")
with mlflow.start_run() as run:
    history = model.fit(X_train_transformed,y_train,
                        epochs=10,
                        validation_data=(X_val_transformed,y_val),
                        verbose=1)

    results = model.evaluate(X_test_transformed,y_test)
    mlflow.log_artifact(filename)
    mlflow.log_metric("test_loss", results[0])
    mlflow.log_metric("test_accuracy", results[1])
print("test loss, test acc:", results)
compare(run.info.run_uuid,name='text')  
