# Neural Network FaceRecognition

This repository contains the source code of a university project of Digital Systems course at University of Bologna. <br>
This project aims to design an Android app using a neural network that can recognizes the face of the user. <br>
The system uses a neural network model trained for recognizing the face of the user in a pic with several people. <br>


## The application 

The central part of the project is an Android application that allows to perform a real-time live face recognition detection with the phone camera and also the possibility of a static filtering of the pics in gallery.


Example of the static filtering:
<p align="center">
   <img src="github images/Immagine 2022-10-29 155835.png">
</p>

Example of live detection:
<p align="center">
   <img src="github images/Immagine 2022-10-29 160035.png">
</p>

## Neural network training

The neural network was trained using Tensorflow/Keras with MobileNet as the starting model. 
<p align="center">
   <img src="github images/reteNeurale.png">
</p>

First of all, we have to upload a good amount of user pictures we want to recognize in the datasets/face_dataset_train_images/me folder. <br>
After that, we have to upload a very huge amount of generic pictures of differen people in the datasets/face_dataset_train_images/not_me folder. <br>
Then, we have to start the script me_not_me_classifier.py in order to obtain a .h5 trained model ready to use located in /models folder. <br>

Using converter.py we can obtain a lighter model (.TFLite) that can be used in mobile systems converted from an .h5 model.

## Python helper tool
We have also developed some helper scripts in Python to test and improve our work. <br>
- data_augmentation.py -> Data augmentation is the process of artificially increasing a dataset size by applying some transformations to the original images  <br>
- downloaderPhotos.py -> this script downloads all the pictures from a CSV dataset and saves them in a local directory  <br>
- convertiImg.py -> this script converts all images in a folder in .PNG format  <br>
- webcam.py -> this script perform a live face detection using the before-trained model with PC camera <br>
- face&resize.py -> this script finds all the faces in the pictures and saves the images of the faces in a standars 250x250 pixel format
