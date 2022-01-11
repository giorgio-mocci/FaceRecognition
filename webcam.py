# face detection with mtcnn on a photograph
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
import cv2
import tensorflow as tf

# draw an image with detected objects

path = "datasets/face_dataset_train_images/not_me/"

def draw_image_with_boxes(filename, result_list):
    # load the image
    data = pyplot.imread(path +filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    
    count =0
    print("trovati "+ str(len(result_list)) + "risultati!")
    img = cv2.imread(path + filename)
    for result in result_list:
        # get coordinates
        x, y, w, h = result['box']
        count +=1
        k=0.5
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
        rect = Rectangle((start_x, start_y), end_x, end_y, fill=False, color='red')
        
        
        crop_img = img[start_y:start_y+end_y , start_x:end_x]
        #crop_img = cv2.resize(crop_img, (252,250), interpolation=cv2.INTER_LINEAR)
        #cv2.imshow("cropped", crop_img)
        filename = "face_" + str(count) + filename 
       # print(path + filename)
       # cv2.imwrite(filename +"face:"+ str(count), crop_img)
        cv2.imwrite(path + filename, crop_img)
        
        # draw the box
        ax.add_patch(rect)
    # show the plot
    #pyplot.show()
 
 
files = os.listdir(path)
print("trovati in totale = " + str(len(files))) 
contatore = 0
 # create the detector, using default weights
detector = MTCNN()
for filename in files:
    
    
     # load image from file
     pixels = pyplot.imread(path + filename)
    
     # detect faces in the image
     faces = detector.detect_faces(pixels)
     # display faces on the original image
     draw_image_with_boxes(filename, faces)
     os.remove(path + filename)
     contatore +=1
     print(contatore)



