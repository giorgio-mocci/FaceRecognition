#!/usr/bin/env python
# coding: utf-8


#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
# ## Import libraries and Setup
import os
import json
from matplotlib import pyplot
import tflite_runtime.interpreter as tflite
import tensorflow as tf
from tensorflow import keras

# Common imports
import numpy as np
from mtcnn.mtcnn import MTCNN
from os.path import dirname, join
import time
import cv2


 def compute(filename, result_list):

        found = false

    # load the image
        data = pyplot.imread(filename)
    # plot the image
        pyplot.imshow(data)
    # get the context for drawing boxes
        ax = pyplot.gca()
    # plot each box

        print("esamino " + str(filename))
        img = cv2.imread(filename)

        if len(result_list) <= 0:
            print("NON SONO STATE TROVATE FACCE")
            return "NO"

        for result in result_list:
            # get coordinates
            x, y, w, h = result['box']
            k = 1
            if x - k * w > 0:
                start_x = int(x - k * w)
            else:
                start_x = x
            if y - k * h > 0:
                start_y = int(y - k * h)
            else:
                start_y = y
            end_x = int(x + (1 + k) * w)
            end_y = int(y + (1 + k) * h)

            # create the shape
            face_image = img[start_y:end_y, start_x:end_x]
            face_image = tf.image.resize(face_image, [250, 250])
            face_image = np.expand_dims(face_image, axis=0)

            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # invoke model with image tensor
            input_shape = input_details[0]['shape']
            input_data = face_image
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            #obtain result for current face
            result = interpreter.get_tensor(output_details[0]['index'])
            prediction = class_names[np.array(result[0]).argmax(axis=0)]  # predicted class
            confidence = np.array(result[0]).max(axis=0)  # degree of confidence

            if prediction == 'me':
                print("SI PORCODIO")
                found = true

            else:
                print("NO PORCODIO")



    if found == true
        return 'SI'
    else
        return 'NO'


def main():

# ## Load Model
# Load model to face classification
    nomeFile = join(dirname(__file__), "model.tflite")
    interpreter = tflite.Interpreter(model_path=nomeFile)
    interpreter.allocate_tensors()
    class_names = ['me', 'not_me']

# create the face locator, using default weights
    detector = MTCNN()
    numeroTrovati = 0
    directory_in_str = '/data/data/com.example.dolgio/app_files'
    directory = os.fsencode(directory_in_str)
    result = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        nome = directory_in_str+"/"+filename
        print(nome)
        if nome.endswith(".jpg"):
            pixels = pyplot.imread(nome)
            faces = detector.detect_faces(pixels)
            print(".jpg")
            if compute(nome, faces) == 'SI':
                print("append")
                result.append(filename)


    print(result)
    y = json.dumps(result)



    return y
