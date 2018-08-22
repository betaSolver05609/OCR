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
import numpy as np

PATH="C:\\Users\\Dell inspiron\\Desktop\\CurrentWorkingDirectory\\OCR\\test_data"

os.chdir(PATH);
Image=cv2.imread('cropped.png');
I=Image.copy();
g=np.zeros(I.shape);
img = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img,(11,11),0)
th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#plt.imshow(th, cmap=plt.cm.gray)
#plt.show();
img=cv2.bitwise_not(th, th)
#plt.imshow(img, cmap=plt.cm.gray)
#plt.show();

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
        areas.append(cv2.contourArea(contour))

median=st.median(areas)

def secSortKey(p):
        return p[0]
def sortKey(p):
        return p[1]

for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)
        if h*w>median:
                a.append(contour);
                cv2.rectangle(I, (x, y), (x + w, y + h), (255, 10, 0), 0)
                        
extracted_image=list();
temp=list();
for contour in a:
        [y,x,w,h]=cv2.boundingRect(contour)
        temp.append([x,y,w,h])

temp.sort(key=sortKey)
temp.sort(key=secSortKey)
for elem in temp:
        [x,y,w,h]=elem
        new_image=img[y:y+h, x:x+w]
        extracted_image.append(new_image)

"""new_temp=np.array(temp)
max_width=np.sum(new_temp[::, (0,2)], axis=1).max()
max_height=np.max(new_temp[::, 3])
nearest=max_height*1.4
temp.sort(key=lambda r: (int(nearest*round(float(r[1])/nearest))*max_width+r[0]))
"""
#cv2.imshow('Image', I);
#cv2.waitKey(1000)
#cv2.imwrite('ContourDetected.png', I)



os.chdir("C:\\Users\\Dell inspiron\\Desktop\\CurrentWorkingDirectory\\OCR\\results")

f=open('output.txt');
ch=f.read();
i=0;
for e in temp:
        [x,y,w,h]=e
        if i<len(ch):
                cv2.putText(g, ch[i], (y,x+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                i=i+2

cv2.imshow('Image', g)
cv2.imwrite('PuttingBack.png', g)

"""def ProcessImage():
        os.chdir(PATH);
        Image=cv2.imread('cropped.png');
        I=Image.copy();
        img = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img,(11,11),0)
        th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        img=cv2.bitwise_not(th, th)
        image, contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS )
        count=0
        a=list();
        areas=list();
        width=list();
        height=list();
        extracted_image=list();
        for contour in contours:
                areas.append(cv2.contourArea(contour))
                median=st.median(areas)
                for contour in contours:
                        [x, y, w, h] = cv2.boundingRect(contour)
                        if h*w>median:
                                cv2.rectangle(I, (x, y), (x + w, y + h), (255, 10, 0), 0)
                                for contour in a:
                                        [x, y, w, h] = cv2.boundingRect(contour)
                                        new_image=img[y:y+h, x:x+w]
                                        extracted_image.append(new_image)
                                        print("Yahan AYA");
        return extracted_image

"""
"""class ProcessImage(object):
        def __init__(self):
                pass;
                #something
        def processImage(self):
                os.chdir(PATH);
                Image=cv2.imread('cropped.png');
                I=Image.copy();
                img = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(img,(11,11),0)
                th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
                img=cv2.bitwise_not(th, th)
                image, contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS )
                count=0
                a=list();
                areas=list();
                width=list();
                height=list();
                extracted_image=list();
                for contour in contours:
                        areas.append(cv2.contourArea(contour))
                        median=st.median(areas)
                        for contour in contours:
                                [x, y, w, h] = cv2.boundingRect(contour)
                                if h*w>median:
                                        cv2.rectangle(I, (x, y), (x + w, y + h), (255, 10, 0), 0)
                                        for contour in a:
                                                [x, y, w, h] = cv2.boundingRect(contour)
                                                new_image=img[y:y+h, x:x+w]
                                                extracted_image.append(new_image)
                                                print("Yahan AYA");
                return extracted_image

""""""G_Image = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
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

#img = cv2.drawContours(Image, contours, -1, (0,255,0), 3)
        

