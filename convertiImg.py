import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import tensorflow as tf
from PIL import Image




path = "C:/Users/gmocc/Documents/GitHub/ProgettoDigitali/provaPython/faceRecognition/datasets/face_dataset_train_images/prova/"
files = os.listdir(path)
for filename in files:
     print("esamino "+ str(filename))
     im = Image.open(path + filename)
     im.save(path + filename + ".png")
