# -*- coding: utf-8 -*-
"""
Created on Fri Sep 08 11:59:55 2017

@author: Insane
"""

import cv2
import numpy as np
from PIL import Image

img = cv2.imread(r'D:\Profiledskin\test\4.jpg',0)

thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#idx =0 
c = max(contours, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(c)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,150,0),2)
mat = Image.fromarray(img)
mat.show()





