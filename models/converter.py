import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import tensorflow as tf

keras_model = tf.keras.models.load_model("face_classifier.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)


tflitemodel= converter.convert()

with open("model.tflite" , "wb" ) as f:
	f.write(tflitemodel)