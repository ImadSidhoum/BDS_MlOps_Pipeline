import tensorflow as tf
import sys, os

converter = tf.lite.TFLiteConverter.from_saved_model(sys.argv[-1]) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)