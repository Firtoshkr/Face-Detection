#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import os 
import numpy as np


# In[6]:


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("C:\\Users\\firto\\Face_data\\haarcascade_frontalface_alt.xml")
face_data = []
skip = 0
face_section = np.zeros((100,100),dtype = "uint8")
dir_path = "C:\\Users\\firto\\Face_data"
name = input("Please Enter Your Name : ")
while True:
    ret, frame = cap.read()
    if ret is False:
        continue;
    gray_Scale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_Scale,1.3,5)
    faces = sorted(faces,key = lambda f:f[2]*f[3])
    for face in faces[-1:]:
        x,y,w,h = face
        offset = 10
        face_section = gray_Scale[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        print(face_section.shape)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(200,0,0),5)
    cv2.imshow("camera",frame)
    if skip%10 == 0:
        face_data.append(face_section)
        
    keypressed = cv2.waitKey(1) & 0xff
    if keypressed == ord('q'):
        break
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
np.save(dir_path + "\\" + name +".npy",face_data)
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




