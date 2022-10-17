import os
import sys
from multiprocessing import process
import mediapipe as mp
import cv2
import numpy as  np
import shutil
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
sys.path.append("../")

os.mkdir('./Cropped_Face_Data')


#This function import imgaes directroy wise and crop them into new directory
def ImportAndCrop():
    try:
        #Get names of each directory and files in it and store them in dictonary
        dict={}
        for root, dirs, files in os.walk(".\Face_Dataset", topdown=True):
            if not dirs:
                dict[root]=files
        with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
        ) as face_detection:
            for root in dict:
                class_num=os.path.split(root)[1]
                os.mkdir(f'./Cropped_Face_Data/{class_num}')

                #Detect all the faces in the image
                for file in dict[root]:
                    image=cv2.imread(os.path.join(root,file))
                    h, w, _ = image.shape
                    x_max = 0
                    y_max = 0
                    x_min = h
                    y_min = w
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    img_h, img_w, _ = image.shape

                    #choose the face which has largest width and full face is inside image
                    if results.detections:
                        max_width=0
                        for detection in results.detections:
                            x=detection.location_data.relative_bounding_box.xmin
                            y=detection.location_data.relative_bounding_box.ymin
                            height=detection.location_data.relative_bounding_box.height
                            width= detection.location_data.relative_bounding_box.width
                            if x>=0 and y>=0 and height>0 and  width>max_width:
                                max_width=width
                                max_one=detection

                        #crop the image
                        x_min=min(x_min,max_one.location_data.relative_bounding_box.xmin)
                        y_min=min(y_min,max_one.location_data.relative_bounding_box.ymin)
                        height=max(x_max,max_one.location_data.relative_bounding_box.height)
                        width=max(y_max,max_one.location_data.relative_bounding_box.width)
                        x_1 = int((x_min)*img_w)
                        x_2 = int((x_min + width)*img_w)
                        y_1 = int((y_min)*img_h)
                        y_2 = int((y_min + height)*img_h)
                    roi = image[y_1:y_2, x_1:x_2]

                    #save the image
                    try:
                        cv2.imwrite(f'.//Cropped_Face_Data/{class_num}/{file}',img = roi )
                    except:
                        print(class_num, file)
                        continue
    except:
        shutil.rmtree('./Cropped_Face_Data')

if __name__=='__main__':

    ImportAndCrop()