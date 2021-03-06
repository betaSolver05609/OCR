# -*- coding: utf-8 -*-
"""
Created on Fri Aug 03 18:56:05 2018

@author: Dell inspiron
"""

import os;
import cv2;
import matplotlib.pyplot as plt;
from matplotlib import path;
import statistics as st

PATH="C:\\Users\\Dell inspiron\\Desktop\\CurrentWorkingDirectory\\OCR\\test_data"

os.chdir(PATH);
Image=cv2.imread('croppedCursive.png');
I=Image.copy();
img = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img,(11,11),0)
th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
plt.imshow(th, cmap=plt.cm.gray)
plt.show();
img=cv2.bitwise_not(th, th)
plt.imshow(img, cmap=plt.cm.gray)
plt.show();

"""G_Image = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
I=Image.copy();
i=Image.copy(); 
p=PreprocessingEngine(Image)
img=p.getGrayImage();
img=p.reduceNoise();
img=p.eliminateBackground();
image, contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#Otsu Thresholding
blur = cv2.GaussianBlur(G_Image,(1,1),0)
ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
"""
image, contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS )
#img = cv2.drawContours(Image, contours, -1, (0,255,0), 3)
count=0
a=list();
areas=list();
width=list();
height=list();
for contour in contours:
        [x,y,w,h]=cv2.boundingRect(contour)
        width.append(w)
        height.append(h)
        areas.append(cv2.contourArea(contour))

median=st.median(areas)
width=st.median(width);
height=st.median(height)


        
for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)
        if h*w>median:
                count+=1
                a.append(contour);
                if w>width:
                        cv2.rectangle(I, (x, y), (x + int(width), y + h), (255, 10, 0), 0)
                        w=int(w-width);
                        x=int(x+width);
                        cv2.rectangle(I, (x,y), (x+w, y+h), (255, 10, 0),0)
                else:
                        cv2.rectangle(I, (x, y), (x + w, y + h), (255, 10, 0), 0)
                        

extracted_image=list();
for contour in a:
        [x, y, w, h] = cv2.boundingRect(contour)
        new_image=img[y:y+h, x:x+w]
        extracted_image.append(new_image)
        
cv2.imshow('Image', I);
cv2.imwrite('ContourImageDetectedForCursive.png', I)



