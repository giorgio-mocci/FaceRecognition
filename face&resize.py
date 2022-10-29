import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

from mtcnn import MTCNN
import cv2

def crop_image(source_dir, dest_dir, mode):
    if os.path.isdir(dest_dir)==False:
        os.mkdir(dest_dir)
    detector = MTCNN()
    source_list=os.listdir(source_dir)
    uncropped_file_list=[]
    for f in source_list:
        f_path=os.path.join(source_dir, f)
        dest_path=os.path.join(dest_dir,f)
        img=cv2.imread(f_path)
        data=detector.detect_faces(img)
        if data ==[]:
            uncropped_file_list.append(f_path)
        else:
            if mode==1:  #detect the box with the largest area

                print(str(data[0]['box']))
                x, y, w, h = data[0]['box']
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
                face_image= img[start_y:end_y,start_x:end_x]
                print("CROPPO E SALVO: "+str(f_path) +"\n")
                dim = (250,250)
                face_image = cv2.resize(face_image,dim,interpolation = cv2.INTER_AREA)
                cv2.imwrite(dest_path, face_image)
            else:
                for i, faces in enumerate(data): # iterate through all the faces found
                    box=faces['box']
                    print(box)
                    if box !=[]:
                        # return all faces found in the image
                        box[0]= 0 if box[0]<0 else box[0]
                        box[1]= 0 if box[1]<0 else box[1]
                        cropped_img=img[box[1]: box[1]+box[3],box[0]: box[0]+ box[2]]
                        fname=os.path.splitext(f)[0]
                        fext=os.path.splitext(f)[1]
                        fname=fname + str(i) + fext
                        save_path=os.path.join(dest_dir,fname )
                        dim = (250,250)

                        cv2.imwrite(save_path, cropped_img)
                        img = cv2.imread(str(save_path)+str(cropped_img), cv2.IMREAD_UNCHANGED)
                        print('Original Dimensions : ',img.shape)
                        img = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
                        print('Resized Dimensions : ',resized.shape)
                        cv2.imwrite(save_path, cropped_img)

    return uncropped_file_list




source_dir='face_dataset_train_images/me' # directory with files to crop
dest_dir='face_dataset_train_images/me/resize' # directory where cropped images get stored
uncropped_files_list=crop_image(source_dir, dest_dir,1) # mode=1 means 1 face per image
for f in uncropped_files_list:
    print(f)
