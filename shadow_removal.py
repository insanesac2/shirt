import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import time
from renderlib import *

"""
shadow_removal.py : Generate an illumination invariant grayscale image. Refer paper 
"Shadow Removal Using Illumination Invariant Image Formation" by Mark S Drew.
"""

def f(i,chi):
	I = np.array(chi[:,0]*np.cos(np.deg2rad(i))+chi[:,1]*np.sin(np.deg2rad(i)))
	I1 = np.copy(I) + abs(np.amin(I))
	mini,maxi = np.amin(I1),np.amax(I1)
	I1[np.logical_or(I1<(mini+0.05*maxi) , I1>(0.95*maxi))] = 0
	non_zeros = np.nonzero(I1)[0]
	N = len(non_zeros)
	std = np.std(I)
	r = np.amax(I) - np.amin(I)
	binwidth = 3.5*std*np.power(N,-1.0/3)
	number_bins = (r/binwidth)
	hist = np.histogram(I,bins = int(number_bins))[0]
	return entropy(hist)


def illuminationInvariant(img):

	"""
	Generate an illumination invariant based on minimum entropy

	Input:
	Colour Image

	Returns:
	Illumination Invariant grayscale image.
	"""

	img = cv2.resize(img,(512,682))
	rows,columns = img.shape[:2]
	img = cv2.GaussianBlur(img,(5,5),0)
	b,g,r = cv2.split(img)
	img = cv2.merge((r,g,b))
	img_reshaped = np.float32(np.reshape(img,(rows*columns,3)))
	img_reshaped_mat = np.copy(np.matrix(img_reshaped))
	img_reshaped_mat[img_reshaped_mat == 0] = 1
	geo_mean = np.power((img_reshaped_mat[:,0]*img_reshaped_mat[:,1]*img_reshaped_mat[:,2]),1.0/3)
	addition = img_reshaped_mat/np.matrix(geo_mean).T
	addition1 = np.reshape(np.array(addition),(rows,columns,3))
	chromaticity = np.log(addition)
	chromaticity1 = np.reshape(np.array(chromaticity),(rows,columns,3))
	U1 = [1,1,0]/np.power(2,1.0/2)
	U2 = [1,1,-2]/np.power(6,1.0/2)
	U = np.array([[U1],[U2]])
	chi = np.matrix(chromaticity)*np.matrix((U.T))
	theta = range(1,180)
	ent  = [f(i,chi) for i in theta]
	min_theta = ent.index(np.amin(ent))
	I = chi[:,0]*np.cos(np.deg2rad(min_theta))+chi[:,1]*np.sin(np.deg2rad(min_theta))
	I = cv2.normalize(I, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
	I = np.uint8(np.reshape(I,(rows,columns)))
	show(I)
	return I

if __name__ == '__main__':
	img = cv2.imread('24.jpg')
	result = illuminationInvariant(img)