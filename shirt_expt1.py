# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:04:59 2017

@author: Insane
"""
from renderlib import *
import cv2, time
import numpy as np
from mylib2 import *
from PIL import Image
from scipy.signal import argrelextrema
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage.filters import sobel
from scipy.spatial import distance


def find_points(x0, y0, x1, y1):
        "Bresenham's line algorithm"
        points_in_line = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points_in_line.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points_in_line.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points_in_line.append((x, y))
        return points_in_line


img = cv2.imread(r'D:\Profiledskin\test\12.jpg')

row,col,ch = img.shape

image_resize_factor = 4.78
row_small = int(row/image_resize_factor)
col_small = int(col/image_resize_factor)
image_small = cv2.resize(img,(col_small,row_small))

img = cv2.medianBlur(img,5)

image_MSF = cv2.pyrMeanShiftFiltering(image_small, np.uint((1e-4)*(row_small*col_small)),np.uint((0.5e-4)*(row_small*col_small)))

contrast_image, contrast_image_double, boundary_segments, segments, boundary_segments_AB_values, status = compute_contrast_image(image_MSF)


threshold = cv2.threshold(contrast_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
				# print self.rows, self.columns

threshold = cv2.resize(threshold, (col, row))

threshold = cv2.threshold(threshold,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

threshold_copy = np.copy(threshold)

contours, hierarchy = cv2.findContours(threshold_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros((row, col), np.uint8)
canvas = np.zeros((row,col),np.uint8)
canvas1 = canvas2 = np.zeros((row,col),np.uint8)


cnt = contours[0]
max_area = cv2.contourArea(cnt)

for cont in contours:
    if cv2.contourArea(cont) > max_area:
        cnt = cont
        max_area = cv2.contourArea(cont)

shirt_contour = sorted(contours, key=cv2.contourArea)[-1]

cv2.drawContours(mask, [shirt_contour],0,(255,255,0),-1) #Draw largest contour

cv2.drawContours(canvas, [shirt_contour],0,(255,0,0),1)


cont = shirt_contour[0]

mask_copy = np.copy(mask)

cordinate = np.argwhere(canvas != 0)


perimeter = cv2.arcLength(cnt,True)
epsilon = 0.01*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)




r1,ch1,c1 = approx.shape

img1 = img2 = img


#mask_new, original_rotated = correct_object_orientation(np.copy(img),angle, np.copy(mask))

#mask_new_copy = np.copy(mask_new)

#original_rotated_imwrite = np.copy(original_rotated)

approx = np.reshape(approx,(r1,c1))

temp = []

for app in approx:
    if app[0] > 3*(col/4):
        temp.append(app)
temp = np.array(temp)

if temp[0][0] > temp[1][0]:
    point = temp[0]
else:
    point = temp[1]    
        
approx = np.array(approx)

idx = 0

idx = np.argwhere(approx == point[0])        

idx = idx[0][0].astype('uint8')

approx = approx[0:idx+1]

approx = np.delete(approx,4,axis = 0)

approx = np.reshape(approx,(idx,ch1,c1))

cv2.drawContours(img,approx,-1,(255,255,0),50)



approx_len = approx.shape[0]

cuff_left_outer = approx[0] 
cuff_left_inner = approx[1]
cuff_right_outer = approx[approx_len-1]
cuff_right_inner = approx[approx_len-2]

lower_left = approx[3]
lower_right = approx[approx_len-4]

armpit_left = approx[2]
armpit_right = approx[approx_len-3]

      
approx_hip = distance.euclidean(lower_right,lower_left)      

approx_chest = distance.euclidean(armpit_left,armpit_right)
    
approx_left_sleeve_inner = distance.euclidean(armpit_left,cuff_left_inner)    

approx_right_sleeve_inner = distance.euclidean(armpit_right,cuff_right_inner)    

#orig = draw_color_boundary(lx,ly,img)
#m = refine_above_crotch(bound_copy)

#canvas1 = canvas[0:col/2,0:row/2]
#canvas2 = canvas[col/2:col,0:row/2]
#
#cordinate1 = np.argwhere(canvas1 != 0)
#cordinate2 = np.argwhere(canvas2 != 0)
#
#
slope1 = (lower_left[0][1]-armpit_left[0][1])/(lower_left[0][0]-armpit_left[0][0])
b1 = armpit_left[0][1] - (slope1*armpit_left[0][0])
#
x1 = -b1/slope1
y1 = 0
#
#


slope2 = (lower_right[0][1]-armpit_right[0][1])/(lower_right[0][0]-armpit_right[0][0])
b2 = armpit_right[0][1] - (slope2*armpit_right[0][0])
#
x2 = -b2/slope2
y2 = 0
#

#diff1 = np.float64(cordinate[:,0] - lower_left[0,1]) 
#diff2 = np.float64(cordinate[:,1] - lower_left[0,0])
#
#slope = round(slope,6)
#
#slopes = np.float64(diff1/diff2)
#
#diff3 = np.abs(slopes) - slope
#
#min_idx = np.argmin(diff3)
#slopes = []
#slopes.append(diff2/diff1).astype('float32')

#print cordinate[min_idx][0],cordinate[min_idx][1]

#cv2.line(img,(x2,y2),(armpit_right[0][0],armpit_right[0][1]),(255,0,0),50)

points =  find_points(armpit_left[0][0], armpit_left[0][1],x1,y1)

points = np.array(points)

length = points.shape[0]

points = np.reshape(points,(length,ch1,c1))

cv2.drawContours(canvas1, points,-1,(255,0,0),1)

res = np.logical_and( canvas, canvas1 )

ind_res_tot = np.argwhere(res == True)

for ind in ind_res_tot:
  if (ind[0] < armpit_left[0][1] and ind[1] < col/2):
      ind_res = ind
     
     
cv2.circle(img,(ind_res[1],ind_res[0]),30, (0,0,255), -1)


points2 =  find_points(armpit_right[0][0], armpit_right[0][1],x2,y2)

points2 = np.array(points2)

length2 = points2.shape[0]

points2 = np.reshape(points2,(length2,ch1,c1))

cv2.drawContours(canvas2, points2,-1,(255,0,0),1)

res2 = np.logical_and( canvas, canvas2 )

ind_res_tot2 = np.argwhere(res2 == True)

for ind in ind_res_tot2:
    if (ind[0] < armpit_right[0][1] and ind[1] > col/2):
      ind_res2 = ind

cv2.circle(img,(ind_res2[1],ind_res2[0]),30, (0,0,255), -1)


#mpl.pyplot.imshow(canvas2)


mpl.pyplot.imshow(img)



#fig = plt.figure(figsize=(14, 6))

#sub1 = fig.add_subplot(411) # instead of plt.subplot(2, 2, 1)
#sub1.plot(t1, vertical_length)
#sub2 = fig.add_subplot(412)
#sub2.plot(t1, first_order)
#sub3 = fig.add_subplot(413)
#sub3.plot(t1,second_order)
#sub3 = fig.add_subplot(414)
#sub3.plot(t1,third_order)

#dif1 = vertical_length - first_order
#dif2 = vertical_length - second_order
#dif3 = vertical_length - third_order
#
#sub1 = fig.add_subplot(411) # instead of plt.subplot(2, 2, 1)
#sub1.plot(t1, vertical_length)
#sub2 = fig.add_subplot(412)
#sub2.plot(t1, dif1)
#sub3 = fig.add_subplot(413)
#sub3.plot(t1,dif2)
#sub3 = fig.add_subplot(414)
#sub3.plot(t1,dif3)


##cv2.imshow('m',m)
#cv2.waitKey()
#cv2.destroyAllWindows
#threshold = Image.fromarray(mask)
#threshold.show()
#plt.show()
#cv2.imshow("Contour", canvas)
#k = cv2.waitKey(0)
#
#if k == 27:         # wait for ESC key to exit
#    cv2.destroyAllWindows()