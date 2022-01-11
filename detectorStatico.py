#!/usr/bin/env python
# coding: utf-8

# ## Import libraries and Setup
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
from matplotlib import pyplot
# Common imports
import numpy as np
# TensorFlow imports
# may differs from version to versions
import tensorflow as tf
from tensorflow import keras
# OpenCV
import cv2
#MTCNN
from mtcnn.mtcnn import MTCNN
import time

# Colors to draw rectangles in BGR
RED = (0, 0, 255)
GREEN = (0, 255, 0)

# Path for recognition
#path = "datasets/face_dataset_train_images/not_me/"
path = "provastatica/"

# ## Load Model
# Load model to face classification
# model was created in me_not_me_classifier.ipynb notebook
model_name = 'face_classifier.h5'
face_classifier = keras.models.load_model(f'models/{model_name}')
class_names = ['me', 'not_me']

conteggioSi = 0




def compute(filename, result_list):
    global Occorrenze

    # load the image
    data = pyplot.imread(path +filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box


    print("esamino "+ str(filename))
    img = cv2.imread(path + filename)

    if len(result_list) <= 0:
        print("NO")
        if not filename.startswith("daniel"):
                Occorrenze +=1
    for result in result_list:
        # get coordinates
        x, y, w, h = result['box']

        k=1
        if x - k*w > 0:
            start_x = int(x - k*w)
        else:
            start_x = x
        if y - k*h > 0:
            start_y = int(y - k*h)
        else:
            start_y = y

        end_x = int(x + (1 + k)*w)
        end_y = int(y + (1 + k)*h)


        # create the shape
        face_image= img[start_y:end_y,start_x:end_x]
        face_image = tf.image.resize(face_image, [250, 250])
        face_image = np.expand_dims(face_image, axis=0)
        result = face_classifier.predict(face_image)
        prediction = class_names[np.array(
            result[0]).argmax(axis=0)]  # predicted class
        confidence = np.array(result[0]).max(axis=0)  # degree of confidence

        if prediction == 'me':
            print("SI")
            if filename.startswith("daniel"):
                Occorrenze +=1
        else:
            print("NO")
            if not filename.startswith("daniel"):
                Occorrenze +=1





files = os.listdir(path)
Occorrenze=0

print("ESAMINO LA CARTELLA "+ path)
numeroIniziale = len(files)
FileIniziali =""
 # create the detector, using default weights
detector = MTCNN()
numeroTrovati =0
while True:

    files = os.listdir(path)
    if len(files) != numeroTrovati:
        time.sleep(2)
        numeroTrovati = len(files)
        print("trovato un nuovo file nella directory!")
        for filename in files:
               # print("incremento")
            if filename not in FileIniziali:
                # load image from file
                pixels = pyplot.imread(path + filename)
                # detect faces in the image
                faces = detector.detect_faces(pixels)
                # display faces on the original image
                compute(filename, faces)



        FileIniziali= os.listdir(path)
        percentuale =  (Occorrenze / len(files)) * 100


        print("Risultati analisi : \n" +"Computate "+ str(len(files))+"\ntrovate occorrenze di daniel%: "+ str(percentuale))
# In[ ]:





# In[ ]:
