
import cv2, math, time, scipy
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
from skimage.util import img_as_float
from skimage import color
from skimage.segmentation import slic
from numba import jit, autojit, cuda
from scipy.linalg import inv
from numpy.linalg import norm
from skimage.segmentation import mark_boundaries
from cudalib import *
from renderlib import *
from shadow_removal import illuminationInvariant
from scipy.signal import argrelextrema
from scipy import signal





def refine_above_crotch(bound):
	"""
	Perfom convex hull on pant contour above crotch point
	Inputs: 
			bound (numpy array) : BOundary image of pant

	Returns:
			mask (numpy array) : smooth contour above crotch
	"""

	mask = np.zeros((bound.shape), dtype = np.uint8)
	
	contours = cv2.findContours(bound, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

	largest = sorted(contours, key = cv2.contourArea)

	cnt = largest[-1]

	cv2.drawContours(mask, [ cnt], -1, 255, 1)

	hull = cv2.convexHull(cnt)
	hull = cv2.approxPolyDP(hull,3, True)
	
	mask[:] = 0
	cv2.drawContours(mask, [hull], -1, 255, -1)
	
	return mask




def correct_image_orientation(image):
	"""
	Orient image to upright position.
	Inputs:
			image (numpy array) : Input image
	Returns:
			res (numpy image) : orientation corrected image
	"""
	b,g,r = cv2.split(image)
	res = np.zeros(image.shape, dtype = np.uint8)
	res = cv2.merge((cv2.flip(b.T,1), cv2.flip(g.T,1), cv2.flip(r.T,1)))

	return res



def correct_object_orientation(original,angle, bound):
	"""
	Rotate pant contour to upright position
	Inputs:
			original (numpy array) : input image
			angle (float) : angle throght which image is to rotated
			bound (numpy image) : boundary image
	Returns:
			bound (numpy image) : new boundary image
			original_rotated (numpy image) : rotated version of input image
	"""
	angle= abs(angle)

	rows, columns = original.shape[:2]

	original_rotated = original.copy()

	
	if angle == 0:
		# print 'Exactly  vertical image', '\n'
		# print 'No need of Manipulation', '\n'
		t,bound=cv2.threshold(bound,0,1,cv2.THRESH_BINARY)
	elif angle>0 and angle <=40:
		# print 'left oriented', '\n'
		# print 'Need to rotate image towards right by ', angle, '\n'
		M = cv2.getRotationMatrix2D((columns /2, rows/2),-angle,1)
		bound = cv2.warpAffine(bound,M,(columns,rows))
		# Threshold Rotated boundary 
		t,bound=cv2.threshold(bound,0,1,cv2.THRESH_BINARY)
		original_rotated  = cv2.warpAffine(original,M,(columns,rows))
		original_rotated = cv2.warpAffine(original,M,(columns,rows))
	elif angle<90 and angle >= 50:
		# print 'Right oriented', '\n'
		# print 'Need to Rotate image towards left by ', angle, '\n'
		M = cv2.getRotationMatrix2D((columns /2, rows/2),90-angle,1)
		bound = cv2.warpAffine(bound,M,(columns,rows))
		#Threshold Rotated boundary 
		t,bound=cv2.threshold(bound,0,1,cv2.THRESH_BINARY)
		original_rotated  = cv2.warpAffine(original,M,(columns,rows))
		original_rotated = cv2.warpAffine(original,M,(columns,rows))

	return bound, original_rotated




def find_lower_corners(lx,ly,image_size_tuple):
	"""
	Find four point closest to four image corners.
	Inputs:
			lx (numpy array) : row co-ordinate of boundary
			ly (numpy array) : column co-ordinate of boundary
			image_size_tuple (tuple) : image size 

	Returns:

	"""

	rows, columns = image_size_tuple

	#origins'  co-ordinates
	lower_left_origin=np.array([rows-1,0])
	lower_right_origin=np.array([rows-1,columns-1])
	upper_left_origin=np.array([0,0])
	upper_right_origin=np.array([0,columns-1])

	#distances arrays
	distances_from_lower_left_origin = np.array([])
	distances_from_lower_right_origin = np.array([])
	distances_from_upper_left_origin = np.array([])
	distances_from_upper_right_origin = np.array([])

	#compute distances from all 4 origins
	for ind in range(len(lx)):
		boundary_point=np.array([lx[ind],ly[ind]])
		
		distances_from_lower_left_origin = np.append(distances_from_lower_left_origin , np.linalg.norm((lower_left_origin - boundary_point), ord=2))
		distances_from_lower_right_origin = np.append(distances_from_lower_right_origin , np.linalg.norm((lower_right_origin - boundary_point), ord=2) )

	lower_left_waist_index = np.argmin(distances_from_lower_left_origin)
	lower_right_waist_index = np.argmin(distances_from_lower_right_origin)

	lower_left_waist = np.array([lx[lower_left_waist_index],ly[lower_left_waist_index]])
	lower_right_waist = np.array([lx[lower_right_waist_index],ly[lower_right_waist_index]])

	approximate_waist_length = np.linalg.norm((lower_left_waist - lower_right_waist ), ord=2)

	return  approximate_waist_length


def find_upper_corners(lx,ly,image_size_tuple):
	"""
	Find four point closest to four image corners.
	Inputs:
			lx (numpy array) : row co-ordinate of boundary
			ly (numpy array) : column co-ordinate of boundary
			image_size_tuple (tuple) : image size 

	Returns:

	"""

	rows, columns = image_size_tuple

#	#origins'  co-ordinates
#	lower_left_origin=np.array([rows-1,0])
#	lower_right_origin=np.array([rows-1,columns-1])
#	upper_left_origin=np.array([0,0])
#	upper_right_origin=np.array([0,columns-1])
#
#	#distances arrays
#	distances_from_lower_left_origin = np.array([])
	distances_from_upper_right = np.array([])
	distances_from_upper_left = np.array([])
	upper_midpoint1 = np.array([0,(columns/2)-(columns/15)])
	upper_midpoint2 = np.array([0,(columns/2)+(columns/15)])
 #compute distances from all 4 origins
	for ind in range(len(lx)):
		boundary_point=np.array([lx[ind],ly[ind]])
		
		distances_from_upper_left = np.append(distances_from_upper_left, np.linalg.norm((upper_midpoint1 - boundary_point), ord=2))
		distances_from_upper_right = np.append(distances_from_upper_right, np.linalg.norm((upper_midpoint2 - boundary_point), ord=2) )

	upper_left_waist_index = np.argmin(distances_from_upper_left)
	upper_right_waist_index = np.argmin(distances_from_upper_right)

	upper_left_waist = np.array([lx[upper_left_waist_index],ly[upper_left_waist_index]])
	upper_right_waist = np.array([lx[upper_right_waist_index],ly[upper_right_waist_index]])

	approximate_neck_length = np.linalg.norm((upper_left_waist - upper_right_waist ), ord=2)

	return  approximate_neck_length


def trace_boundary(binary):
	"""
	Extract boundary points' co-ordinates 
	Inputs:
			binary (numpy array) : binary image of object

	Returns:
			lx (numpy array) :row co-ordinate of boundary
			ly (numpy array) :column co-ordinate of boundary
			bound (numpy array) : one pixel thick object boundary


	"""
	time1 = time.time()
	binary = np.uint8(binary)
	contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
	cnt = sorted(contours, key = cv2.contourArea)[-1]
	ly, lx = contour2lxly(cnt) #function returns opencv style co-ordinates

	bound = np.zeros(binary.shape, dtype = np.uint8)
	cv2.drawContours(bound, [cnt], -1,1, 10)
	time2 = time.time()
	return lx, ly, bound





def draw_color_boundary(lx,ly, original):
	"""
	Draw color boundary from given points
	Inputs:
			lx (numpy array) :row co-ordinate of boundary
			ly (numpy array) :column co-ordinate of boundary
			original (numpy array) : input image
	Returns:
			original (numpy array) : input image with object boundary highlighted

	"""
	lx = np.uint32(lx)
	ly = np.uint32(ly)
	original[lx,ly,0] = 0
	original[lx,ly,1] = 255
	original[lx,ly,2] = 0
	return  original



def harris(binary, approximate_waist_length, no_of_corners, offset, hull_flag):
	"""
	Detect corner points with given parameters
	Inputs:
		binary (numpy array) : binary object boundar image 
		approximate_waist_length (float) : approx. waist length
		no_of_corners (int) : number of corners
		offset (int) : offset value
		hull_flag (boolean) : flag for performing convex hull

	Returns:
			corners (list) : co-ordinates of corners
	"""

	binary_org = np.copy(binary)

	corners = cv2.goodFeaturesToTrack(binary,no_of_corners,0.1,approximate_waist_length/4.0)
	corners = np.int0(corners)

	for i in corners:
		x_corner,y_corner = i.ravel() #opencv style
		column, row = i.ravel() #array style
		# cv2.circle(original,(x_corner,y_corner+offset),30,(0,0,255),1)
		# cv2.circle(binary_org ,(x_corner,y_corner),30,255,1)
	return corners



def lengthOfContour(contour, imshow_flag = True):
	return len(contour)




def find_note(note_slice, image_name):
	"""
	Localise note inside note_slice region. Mark corners of note
	Inputs:
			note_slice (numpy array) : image slice containing note
			image_name (str) : name of the input image

	Returns:
			corners (list) : corners of note


	"""
	rows, columns = note_slice.shape[:2]
	note_slice_org = np.copy(note_slice)
	
	note_slice = cv2.bilateralFilter(note_slice,9,75,75)
	hsv_note_slice = cv2.cvtColor(note_slice, cv2.COLOR_BGR2HSV)
	
	gray_image = cv2.cvtColor(note_slice,cv2.COLOR_BGR2GRAY)
	
	threshold = np.zeros(gray_image.shape, dtype = np.uint8)
	ret,threshold = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #1
	

	threshold/=255
	

	sel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	
	threshold = cv2.erode(threshold, sel, iterations = 2)
	



	contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	largest = sorted(contours, key = cv2.contourArea)
	cnt = largest[-1]
	mask = np.zeros((rows, columns), dtype = np.uint8)
	
	x,y,w,h = cv2.boundingRect(cnt)
	
	
	# mask = np.zeros((w,h), dtype = np.uint8)
	if x-10>0 and w+20 < columns-1:
		x-=10
		w+=20
	
	if y-10 >0 and h+20 < rows-1:
		y-=10
		h+=20
	

	
	note_slice = note_slice[y:y+h,x:x+w]
	

	gray_image = cv2.cvtColor(note_slice,cv2.COLOR_BGR2GRAY)
	
	ret,threshold = cv2.threshold(gray_image,ret,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #1
	

	contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	largest = sorted(contours, key = cv2.contourArea)
	cnt = largest[-1]
	hull = cv2.convexHull(cnt)
	res = np.zeros(threshold.shape, dtype = np.uint8)
	cv2.drawContours(res, [hull], -1, (255),1)
	cv2.drawContours(note_slice, [hull], -1, (255,0,0),1)
	# show(note_slice, 'note_contour', render = True)

	lx,ly = contour2lxly(hull)
	lx+=x
	ly+=y

	hull = lxly2contour(ly,lx)
	cv2.drawContours(note_slice_org, [hull], -1, (255,0,0),1)
	
	no_of_corners = 4
	corners = cv2.goodFeaturesToTrack(res,no_of_corners,0.1,50)
	corners = np.int0(corners)
	
	return corners







def pertinent_background(labels, segments_median_value, pixel_count_for_labels, side):
	"""
		Find out a label which has least distance along the given image boundary
		Inputs:
				labels (numpy array) : SLIC labels image
				segments_median_value (numpy array) : median values for each label
				pixel_count_for_labels (numpy array) : count of pixels in each superpixel label
				side (int) : indicates which boundary of image is considered

		Returns:
				pertinent_background_index (int) : index of pertinent background.
	"""
	
	rows, columns = labels.shape
	
	if side == 0:
	
		boundary_segments = np.copy(labels[:,0])
	elif side ==1:
	
		boundary_segments = np.copy(labels[rows-1,:])
	elif side == 2:
	
		boundary_segments = np.copy(labels[:,columns-1])
	elif side == 3:
	
		boundary_segments = np.copy(labels[0,:])
	

	unq_boundary_segments = np.unique(boundary_segments) 
	
	e0 = cv2.getTickCount()
	contrast = np.zeros(len(unq_boundary_segments))
	for i in range(len(unq_boundary_segments)):
		temp = 0
		for j in range(len(unq_boundary_segments)):
				temp+= np.sum((segments_median_value[unq_boundary_segments[i]] / segments_median_value[unq_boundary_segments[j]])**2, axis = None) #+ (segments_median_value[boundary_segments[i],2] - segments_median_value[boundary_segments[j],2])**2

		contrast[i] = temp 

	
	e1 = cv2.getTickCount()
	
	pertinent_background_index = boundary_segments[np.where(contrast==np.min(contrast))].ravel()[0]
	
	return pertinent_background_index


def canny_function(image):
	"""
	canny edge detector
	Inputs:
			image (numpy array) : grayscale image
	Returns:
			edged (numpy array) : edge image
	"""
	# sigma=0.01
	v = np.median(image)
	lower = np.amin(image)#0#int(max(0, (1.0 - sigma) * v))
	upper = np.amax(image)#255#int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	return edged



def offset_kernel(binary, patch_offset):
	"""
	This function adds offset to change patch dimensions.
	THIS FUNCTION IS NOT BEING USED CURRENTLY. IT IS DEPRECATED. 

	Inputs:
			binary (numpy array) : binary image
			patch_offset (int) : offset for given patch

	Returns:
	offset_left 	(int) 	: offset value for left boundary of image
	offset_right 	(int) 	: offset value for right boundary of image
	offset_up 		(int) 	: offset value for up boundary of image
	offset_down 	(int) 	: offset value for down boundary of image
	"""
	rows, columns = binary.shape
	offset_up = 0
	offset_down = 0 
	offset_left = 0
	offset_right = 0

	xs,ys = np.nonzero(binary)
	x, y = int(np.mean(xs)), int(np.mean(ys))

	dist1 = y #distance form left edge
	dist2 = x #distance from top edge
	dist3 = columns - y#dist from right edge
	dist4 = rows - x#dist from bottom edge

	if dist1<=dist3:
		offset_left = patch_offset
	else:
		offset_right = patch_offset

	if dist2<=dist4:
		offset_up = patch_offset
	else:
		offset_down = patch_offset

	return (offset_left, offset_right, offset_up, offset_down)


def get_offset(binary, area_prev):
	"""
	This function determines what offset to be added to get desired results.
	THIS FUNCTION IS NOT BEING USED CURRENTLY. IT IS DEPRECATED. 

	Inputs:
			binary 		(numpy array)	: binary input patch
			area_prev 	(int)			: area in previous step

	Returns:
			offset_left 	(int) 	:	offset value for left boundary of image
			offset_right 	(int) 	:	offset value for right boundary of image
			offset_up 		(int) 	:	offset value for up boundary of image
			offset_down 	(int) 	:	offset value for down boundary of image
			non_zeros_len	(int)	:	number of non-zero pixels
	"""

	# print  area_prev
	rows, columns = binary.shape
	non_zeros_len = len(np.nonzero(binary)[0])

	area = rows*columns
	zeros_len = area - non_zeros_len
	ratio = non_zeros_len*1.0/(area*1.0)
	# print 'ratio NZ/Area', ratio
	if ratio >=0.4 and ratio<=0.6:#non_zeros_len>0.4*zeros_len and non_zeros_len<0.6*zeros_len:
		# print 'Ratio is OK. Exit current patch'
		return [[-1,-1,-1,-1], -1]

	
	if zeros_len>non_zeros_len:
		# print 'Case1!'
		offset_left, offset_right, offset_up, offset_down = offset_kernel(binary, (1.0/abs(0.5-ratio))) #offset_kernel(binary, (1.0/abs(0.5-ratio)))
		
	else:
		# print 'case 2!'
		binary = (binary-1)/255
		offset_left, offset_right, offset_up, offset_down = offset_kernel(binary, (1.0/abs(0.5-ratio)))#offset_kernel(binary, (1.0/abs(0.5-ratio)))

	
	return [[offset_left, offset_right, offset_up, offset_down], non_zeros_len]




def cuff_corner(binary, binary_org,offset_x, offset_y):
	"""
	This function determines cuff corners from binary image

	Inputs:
			binary 		(numpy array)	:	slice of binary image
			binary_org	(numpy array)	:	complete binary image
			offset_x 	(int) 			:	offset along x direction
			offset_y	(int)			:	offset along y direction

	Returns:
			cuff_points	(numpy array)	:	cuff point co-ordinates
	"""
	m,n = binary.shape

	
	res = np.zeros((m,n),dtype = np.uint8)

	
	ly,lx = contour2lxly(sorted(cv2.findContours(np.copy(binary), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0], key = cv2.contourArea)[-1])
	
	lx = lx[0:len(lx)/2]
	ly = ly[0:len(ly)/2]
	x_points = lx[lx>(np.amax(lx)-150)]
	y_points = ly[lx>np.amax(lx)-150]
	y_points_small_right = y_points[y_points>np.amax(y_points) - 100]
	x_points_small_right = x_points[y_points>np.amax(y_points) - 100]
	y_points_small_left = y_points[y_points<np.amin(y_points) + 100]
	x_points_small_left = x_points[y_points<np.amin(y_points) + 100]
	
	res[x_points_small_right,y_points_small_right] = 255
	res[x_points_small_left,y_points_small_left] = 255

	rightmost_left_y = max(y_points_small_left)
	rightmost_left_x = x_points_small_left[np.argmax(y_points_small_left)]

	leftmost_right_y = min(y_points_small_right)
	lestmost_right_x = x_points_small_right[np.argmin(y_points_small_right)]

	cv2.line(res,(rightmost_left_y,rightmost_left_x),(leftmost_right_y,lestmost_right_x),255,1)
	# show(res, 'res')
	final_contour,hier = cv2.findContours(np.copy(res), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	res1 = np.zeros(res.shape, dtype = np.uint8)
	hull = cv2.convexHull(final_contour[0])
	cv2.drawContours(res1, [hull], -1, (255), 1)
	# show(res1, 'cnt')

	x_ = max(x_points_small_left[np.argmin(x_points_small_left)], x_points_small_right[np.argmin(x_points_small_right)]) + 5
	dist = np.linalg.norm(np.array([x_points_small_left[0], y_points_small_left[0]]) - np.array([x_points_small_right[-1], y_points_small_right[-1]]), ord=2)
	# print 'dist', dist
	res3 = res1[x_:,:]
	# show(res3, 'res3')
	corners = cv2.goodFeaturesToTrack(res3.copy(),2,0.1,dist/2)

	# final_res = np.zeros(binary.shape)
	cuff_points = np.zeros((2,2), dtype = np.int32)
	corners = np.int0(corners)
	for index,i in enumerate(corners):
		x_corner,y_corner = i.ravel() #opencv style
		column, row = i.ravel() #array style
		# #print 'harris',[row, column+offset], (np.where(np.all(bound_zip==[row, column], axis = 1)==True)[0]) 
		# print x_corner + offset_x,y_corner + offset_y
		cv2.circle(res3 ,(x_corner ,y_corner ),10,255,1)
		cv2.circle(binary_org ,(x_corner + offset_x  ,y_corner + offset_y + x_ ),10,255,1)
	# show(res3)
		# print x_corner + offset_x  ,y_corner + offset_y + x_
	# show(binary_org, 'org '+ str(index))
	if corners[0].ravel()[0] > corners[1].ravel()[0]:
		cuff_points[0,0], cuff_points[0,1] = corners[1].ravel()
		cuff_points[1,0], cuff_points[1,1] = corners[0].ravel()
	else:
		cuff_points[0,0], cuff_points[0,1] = corners[0].ravel()
		cuff_points[1,0], cuff_points[1,1] = corners[1].ravel()

	cuff_points[:,0]+= offset_x
	cuff_points[:,1]+= offset_y + x_

	return cuff_points





def improve_boundary(lx, ly, original,crotch,patch_size = (16,12)):
	"""
	This function iterates over all boundary points to improve boundary by thresholding over a small window 
	around each boundary point.
	
	Inputs:
			lx 				(numpy array)	:	row co-ordinates of boundary points
			ly 				(numpy array)	:	column co-ordinates of boundary points
			original 		(numpy array)	:	original image
			crotch 			(tuple)			:	crotch (row, column) co-ordinates
			patch_size 		(tuple)			:	patch size (rows, columns)

	Returns:
			lx 				(numpy array)	:	row co-ordinates of boundary points()
			ly 				(numpy array)	:	column co-ordinates of boundary points()
			bound 			(numpy array)	:	boundary image
			bound_filled	(numpy array)	:	flood filled boundary

			IF OBJECT BOUNDARY TOUCHES IMAGE BOUNDARY, IMPROVEMENT CAN NOT BE OERFORMED.
			IN THAT CASE (-2, -2, -2, -2) IS RETURNED TO INDICATE THE EXCEPTION
	"""
	time1 = time.time()
	crotch_row, crotch_column = crotch
	patch_size_row, patch_size_column = patch_size
	# original = cv2.GaussianBlur(original, (5,5), 0)
	original_copy = np.copy(original)

	rows, columns = original.shape[:2]

	cnt = lxly2contour(lx, ly)

	mask = np.zeros((rows, columns), dtype = np.uint8)
	bound = np.zeros((rows, columns), dtype = np.uint8)

	cv2.drawContours(mask, [cnt], -1, (1), -1)

	# show(mask, 'mask', render = imshow_flag)

	sel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))


	b,g,r = cv2.split(original)

	crop = cv2.merge((b*mask, g*mask, r*mask))

	ly,lx = contour2lxly(sorted(cv2.findContours(np.copy(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0], key = cv2.contourArea)[-1])
	
	# patch_size_row = 16
	# patch_size_column = 12


	gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
	res = np.zeros((rows, columns), dtype = np.uint8)

	res2 = np.zeros((rows, columns), dtype = np.uint8)
	bound_filled = np.zeros((rows, columns))

	flag_pant_touching_border = False

	time1 = time.time()

	ret_global = 0; count = 0
	for i in range(0,len(lx),int((patch_size_row+patch_size_column)/4.0)):
		

		flag_stop = False

		x = lx[i]
		y = ly[i]

		center_x, center_y = x, y

		top_left_row = center_x - patch_size_row/2
		top_left_column = center_y - patch_size_column/2
		bottom_right_row  = center_x + patch_size_row/2
		bottom_right_column  = center_y + patch_size_column/2 

		offset_up_patch = 0;  offset_down_patch = 0; 	offset_left_patch = 0; 	offset_right_patch = 0

		area_prev = -1

		while((flag_stop or flag_pant_touching_border) == False):
			top_left_row_temp = top_left_row - offset_up_patch
			top_left_column_temp = top_left_column - offset_left_patch 
			bottom_right_row_temp  = bottom_right_row + offset_down_patch 
			bottom_right_column_temp  = bottom_right_column + offset_right_patch


			row_indices = np.array([top_left_row_temp, bottom_right_row_temp ])

			column_indices = np.array([top_left_column_temp, bottom_right_column_temp])

			if np.any(row_indices<=0) or np.any(row_indices>=rows-2) or np.any(column_indices<=0) or np.any(column_indices>=columns-2):
				print 'Negative/Out of bound indices encountered in improve'
				flag_pant_touching_border = True
				continue

			else:
				top_left_row = int(top_left_row_temp)
				top_left_column = int(top_left_column_temp)
				bottom_right_row = int(bottom_right_row_temp)
				bottom_right_column = int(bottom_right_column_temp)
			

				patch = original[ top_left_row:bottom_right_row, top_left_column:bottom_right_column ]

				ret, threshold = cv2.threshold(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 0,1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
				ret_global+=ret
				count+=1
				
				res[top_left_row:bottom_right_row, top_left_column:bottom_right_column ] = canny_function(threshold)
				flag_stop = True
				
	ret_global = int(ret_global/ count)

	time2 = time.time()

	if flag_pant_touching_border != True:
		sel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

		res = cv2.dilate(res/255, sel, iterations = 10)

		cv2.drawContours(res2, [ sorted(cv2.findContours(res.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[0], key = cv2.contourArea)[-1]], -1, (1), -1)

		sel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

		res2 = cv2.erode(res2, sel, iterations = 10)

		crotch_patch = original[crotch_row-128 : crotch_row + 128, crotch_column-128 : crotch_column + 128]
		crotch_patch_rows, crotch_patch_columns = crotch_patch.shape[:2]
		threshold = cv2.threshold(cv2.cvtColor(crotch_patch, cv2.COLOR_BGR2GRAY),ret_global ,255,cv2.THRESH_BINARY)[1]
		len_nonzeros = len(np.nonzero(threshold[0,:])) + len(np.nonzero(threshold[-1,:])) + len(np.nonzero(threshold[:,0])) + len(np.nonzero(threshold[:, -1]))
		if len_nonzeros <0.5*(crotch_patch_rows+crotch_patch_columns)*2:
			threshold = abs(threshold-1)/255

		res2[crotch_row-128 : crotch_row + 128, crotch_column-128 : crotch_column + 128] = threshold
		# subplot([crotch_patch, threshold])


		cv2.drawContours(bound, [sorted(cv2.findContours(res2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0], key = cv2.contourArea)[-1]], -1, (1), 1 )
	


		contours = cv2.findContours(np.uint8(bound), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
		
		cnt = sorted(contours, key = cv2.contourArea)[-1]

		cv2.drawContours(original, [cnt], -1, (255,0,0), 1)
	
		cv2.drawContours(bound_filled, [cnt], -1, (1), -1)
		lx, ly, bound = trace_boundary(bound_filled)



		return lx,ly, bound, bound_filled
	else:
		return -2, -2, -2, -2



def fourier_descriptor(contour, rows, columns):
	"""
	This function finds fourier descriptors for given image. Reducing the number of points for fft results in smoother boundary.

	Inputs:
			contour 	(list)	:	contour co-ordinates as returned by OpenCV
			rows 		(int)	:	number of rows
			columns 	(int)	:	number of columns
	"""

	lx, ly = contour2lxly(contour)
	data = lx + 1j * ly;
	data_fft = np.fft.fft(data[::100]) 

	recovered = np.fft.ifft(data_fft)
	lx_new = np.real(recovered)
	ly_new = np.imag(recovered)
	new_cnt = lxly2contour(ly_new, lx_new)
	temp1 = np.zeros((rows, columns),np.uint8 )
	temp2 = np.zeros((rows, columns),np.uint8 )

	cv2.drawContours(temp1, [contour], 0, (255), -1) #Draw largest contour
	cv2.drawContours(temp2, [new_cnt], 0, (255), -1) #Draw largest contour
	# subplot([temp1, temp2])
	return
	



def lxly2contour(lx_boundary, ly_boundary):

	"""
	This function converts two arrays of co-ordinates into a list of paired co-ordinates. 
	It helps to deal with contour functions in OpenCV.
	
	Inputs:
			lx_boundary 	(numpy array)	:  	row  co-ordinates for boundary point
			ly_boundary	 	(numpy array)	:  	column  co-ordinates for boundary point
	Returns:
			contour 		(list)			:	contour co-ordinates in OpenCV's format
	"""
	#Inputs are int form (row, column) for the given point
	contour = [[i,j] for i,j in zip(ly_boundary,lx_boundary)] 
	contour = np.asarray(contour, dtype = np.int32)
	contour = contour
	
	return contour


def contour2lxly(contour):
	"""
	This function extracts boundary points' co-ordinates from contour 
	Inputs:
			contour (list) : contour co-ordinates in OpenCV format

	Returns: 	
			lx (numpy array) : row  co-ordinates for boundary point
			ly (numpy array) : column  co-ordinates for boundary point
	"""
	contour = contour.tolist()
	lx, ly = [], []
	for i in range(len(contour)):
		lx.append(contour[i][0][0])

	for i in range(len(contour)):
		ly.append(contour[i][0][1])

	return np.array(lx),np.array(ly)


def expand_bounding_rect(bound,x,y,w,h,expand_by_length_step = 10):
	"""
	Expand bounding rect on note region to accomodate complete note inside the rectangle
	Inputs:
			bound 					(numpy array)	:	boundary image
			x 						(int)			:	top left row co-ordinate
			y 						(int)			:	top left column co-ordinate
			w 						(int)			:	width of rectangle
			h 						(int)			:	height of rectangle
			expand_by_length_step 	(int)			:	step size for increasing rectangle width and height

	Returns:
			x 						(int)			:	updated top left row co-ordinate
			y 						(int)			:	updated top left column co-ordinate
			w 						(int)			:	updated width of rectangle
			h 						(int)			:	updated height of rectangle

	"""
	max_val = np.max(bound)
	rows, columns = bound.shape
	flag_max_size_reached = False
	count = 0

	while(flag_max_size_reached is not True ):
		#print 'inside while loop'
		if x-expand_by_length_step >0 and w+ 2*expand_by_length_step <columns-1:
			x-=expand_by_length_step
			w+= 2*expand_by_length_step
		if y-expand_by_length_step >0 and h+ 2*expand_by_length_step < rows-1:
			y-=expand_by_length_step
			h+= 2*expand_by_length_step

		if np.all(bound[y:y+h, x:x+w] == max_val):
			count+=1
			continue
		else:
			flag_max_size_reached = True
			x+=expand_by_length_step
			y+=expand_by_length_step
			w-= 2*expand_by_length_step
			h-= 2*expand_by_length_step

	return x,y,w,h

def note_contour_from_hierarchy(contours, hierarchy, largest_contour):
	"""
	Locate contour which has note. This is achieved from contour hierarchy in OpenCV. Pant gives the largest contour. A largest contour inside pant_contour  is note contour

	Inputs:
			contours 			(list)	:	all contours 
			hierarchy 			(list)	:	hierarchy of contours
			largest_contour		(list)	:	largest contour

	Returns:
			note_contour 		(list) 	:	note contour co-ordinates
	"""
	largest_parent_contour_index = contours.index(largest_contour)
	
	mod_cnt = []
	for i in range(len(contours)):
		if hierarchy[0][i][3] == largest_parent_contour_index:
			mod_cnt.append(contours[i])

	largest_cnt = sorted(mod_cnt, key = cv2.contourArea)[-1]
	index = contours.index(largest_cnt)
	note_contour = contours[index]
	return note_contour



def boundary_to_contour_to_approx_bound(lx_boundary, ly_boundary, image_size_tuple):
	"""
	This function smoothes the pant boundary using approxPolyDP function in OpenCV

	Inputs:
			lx_boundary 		(numpy array):	row co-ordinate of boundary
			ly_boundary			(numpy array):	column co-ordinate of boundary
			image_size_tuple	(tuple):		image size 
		
	Returns:
			lx 				(numpy array):	row co-ordinate of boundary
			ly 				(numpy array):	column co-ordinate of boundary
			bound 			(numpy array):	bound image
			bound_filled	(numpy array):	flood filled bound image
	"""
	# Roundness=(4*Area*pi)/(Perimeter.^2) 
	h, w = image_size_tuple
	
	cnt = lxly2contour(lx_boundary, ly_boundary)

	peri = cv2.arcLength(cnt, True)
	roundness_1 = 4*cv2.contourArea(cnt)*np.pi/(cv2.arcLength(cnt, True)**2)
	
	approx = cv2.approxPolyDP(cnt,math.pow(peri,1/np.pi)*2, True)

	roundness_2 = 4*cv2.contourArea(approx)*np.pi/(cv2.arcLength(approx, True)**2)	
	
	mask2 = np.zeros((h,w))
	cv2.drawContours(mask2, [approx],-1,255,-1 )

	bound_filled = mask2
	lx,ly,bound = trace_boundary(bound_filled)

	return lx, ly, bound, bound_filled


def locate_crotch(bound, bound_filled):

	"""
	This function locates crotch in image based on height of vertical columns in binary image.
	Global minima is found using a scipy function argrelextrema

	Inputs:
			bound 			(numpy array):	bound image
			bound_filled	(numpy array):	flood filled bound image
	Returns:
			crotch_column 	(int)	:	column for crotch location
			crotch_row 		(it) 	:	row for crotch location

	"""
	max_val = np.max(bound)
	time1 = time.time()
	rows, columns = bound.shape[:2]

	vertical_length = np.sum(bound_filled, axis = 0)
	# plot(vertical_length)

	
	window = signal.gaussian(20,30)

	vertical_length = signal.convolve(vertical_length,window , mode = 'same')/sum(window)
	vertical_length = signal.convolve(vertical_length,window , mode = 'same')/sum(window)
	vertical_length = signal.convolve(vertical_length,window , mode = 'same')/sum(window)
	
	
	lower = argrelextrema(vertical_length, np.less,order = 100)
	# print 'lower', lower

	locations = lower[0][0]

	crotch_x, crotch_y = None, None

	temp= bound[:,locations][::-1]  #reversing array to find lowermost point
	row_val = np.where(temp==max_val)

	var = (locations,rows-1- row_val[0][0])
	crotch_column, crotch_row = var #numpy style

	time2 = time.time()

	return crotch_column, crotch_row





def compute_contrast_image(image):
	"""
	This function computes SLIC over input image in LAB colorspace. 
	Mean LAB values are calculated using CUDA implementation
	Pertinent background is computed.
	Contrast is calculated for each label using the pertinent background information
	Conrast mapped to an image using CUDA implementation
	
	Inputs:
			image (numpy array)	:	input image

	Returns:
			contrast_image 					(numpy array):	contrast image 
			contrast_image_double			(numpy array):	contrast image in double type
			boundary_segments 				(numpy array):	pertinent boundary segments
			labels 							(numpy array):	labels obtained from SLIC
			boundary_segments_AB_values 	(numpy array):	A, B values for pertinent background segments
			status 									(str):	'success' for successful execution of function


	"""
	status = None
	b,g,r = cv2.split(image)
	image = cv2.merge((r,g,b))

	image_height, image_width = image.shape[:2]
	
	labels = slic(image, n_segments = 20000, sigma = 5, compactness = 30)

	no_of_labels = np.max(labels) + 1
		
	lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


	lab_image_shifted = np.float32(np.copy(lab_image))
	lab_image_shifted[:,:,0]+= 1
	lab_image_shifted[:,:,1] +=   abs(np.min(lab_image_shifted[:,:,1])) + 1 #shifting all values to positive range...easier for doing log operation...to avoid / by 0 add 1
	lab_image_shifted[:,:,2] +=   abs(np.min(lab_image_shifted[:,:,2])) + 1 #shifting all values to positive range...easier for doing log operation ...to avoid / by 0 add 1


	pixel_count_for_labels = np.zeros((no_of_labels,1), dtype = np.uint64)
	cumulative_lab_values = np.zeros((no_of_labels, 3))


	block_width, block_height = image_width,1
	no_of_blocks_x, no_of_blocks_y = image_width/block_width, image_height/block_height

	try:
		time1 = time.time()
		cuda_find_mean_lab[(no_of_blocks_x, no_of_blocks_y), (block_height, block_width )](labels,pixel_count_for_labels, lab_image_shifted, cumulative_lab_values )
		time2 = time.time()
		print 'cuda1', time2-time1
		mean_lab_values = cumulative_lab_values/pixel_count_for_labels
	except:
		status = 'Error in CUDA implementation' 
		return None, None, None, None,None, status



	#Find segment numbers along image boundary
	boundary_segments = np.array([])

	boundary_segments = np.append(boundary_segments, pertinent_background( labels, mean_lab_values, pixel_count_for_labels, side = 3))
	boundary_segments = np.append(boundary_segments, pertinent_background( labels, mean_lab_values, pixel_count_for_labels, side = 1))
	boundary_segments = np.append(boundary_segments, pertinent_background( labels, mean_lab_values, pixel_count_for_labels, side = 0))
	boundary_segments = np.append(boundary_segments, pertinent_background( labels, mean_lab_values, pixel_count_for_labels, side = 2))


	boundary_segments = np.unique(boundary_segments)


	#Compute contrast with assumed background 
	contrast_image = np.zeros((image_height ,image_width), dtype = np.uint8)
	contrast_image_double = np.zeros((image_height,image_width), dtype = np.float32)


	segments_mean_value_AB = np.copy(mean_lab_values[:,1:])
	boundary_segments_AB_values = np.zeros((len(boundary_segments), 2), dtype = np.uint64)
	#Collect boundary segments median values
	time1 = time.time()
	boundary_segments = np.int64(boundary_segments)
	for i in range(len(boundary_segments)):
		boundary_segments_AB_values[i] = segments_mean_value_AB[boundary_segments[i]]

	contrast_array = np.zeros((1,no_of_labels))

	for i in range(len(boundary_segments_AB_values)):
		contrast_array += np.sum(segments_mean_value_AB  /  boundary_segments_AB_values[i] , axis = 1)


	contrast_image_double = np.zeros((image_height ,image_width), dtype = np.float32)

	try:
		time1 = time.time()
		make_contrast_image[(no_of_blocks_x, no_of_blocks_y), (block_height, block_width )](labels, contrast_array, contrast_image_double)
		time2 = time.time()
		print 'cuda2', time2-time1
	except:
		status = 'Error in CUDA implementation2'
		return None, None, None, None,None, status
		
	contrast_image = np.uint8((contrast_image_double)/(np.max(contrast_image_double))*255.0) 

	return contrast_image, contrast_image_double, boundary_segments, labels, boundary_segments_AB_values, 'success'




def find_waist_points(above_crotch, crotch_x, crotch_y, approximate_waist_length, no_of_corners, offset, hull_flag):
	"""
	This function helps in determining Waist points in binary image above crotch location.
	Harris(...) determines corner points as per the given parameters

	Inputs:
			above_crotch 				(numpy array):	boundary image above crotch
			crotch_x 					(int):			crotch row co-ordinate
			crotch_y 					(int):			crotch column co-ordinate
			approximate_waist_length 	(float):		approx waist length
			no_of_corners 				(int):			number of corners
			offset  					(int):			offset 
			hull_flag					(boolean):		flag for performing convex hull

	Returns:
			waist_left_column, waist_left_row 			(int):	co-ordinates for top left waist point
			waist_right_column, waist_right_row 		(int):	co-ordinates for top right waist point
			waist_midpoint_column, waist_midpoint_row	(int):	co-ordinates for midpoint of above waist points
	"""
	waist_corners = harris(above_crotch, approximate_waist_length, no_of_corners, offset, hull_flag)
	#Determine waist co-ordinates
	upper_left_origin=np.array([0,0])
	waist1, waist2 = np.linalg.norm((upper_left_origin - waist_corners[0]), ord=2), np.linalg.norm((upper_left_origin - waist_corners[1]), ord=2)
	if waist1<waist2:
		waist_left, waist_right= waist_corners[0], waist_corners[1]
	else:
		waist_left, waist_right= waist_corners[1], waist_corners[0]

	#For array processing
	waist_left_column = waist_left[0][0]
	waist_left_row = waist_left[0][1]
	waist_right_column = waist_right[0][0]
	waist_right_row = waist_right[0][1]


	#For opencv drawing
	waist_left_x = waist_left[0][0]
	waist_left_y = waist_left[0][1]
	waist_right_x = waist_right[0][0]
	waist_right_y = waist_right[0][1]


	waist_midpoint_column = np.uint16((waist_left_column + waist_right_column)/2.0)
	waist_midpoint_row = np.uint16((waist_left_row + waist_right_row)/2.0)

	# cv2.circle(original_rotated_1,(waist_left_x,waist_left_y),3, (0,0,255), -1)
	# cv2.circle(original_rotated_1,(waist_right_x,waist_right_y),3, (0,255,0), -1)
	# # cv2.circle(original_rotated_1, (crotch_x, crotch_y), 3, (255,0,0), -1)
	# cv2.circle(original_rotated_1,(waist_midpoint_column,waist_midpoint_row),3, (255,255,255), -1)
	# show(original_rotated_1, 'boundary with waist points', render = imshow_flag)


	return waist_left_column, waist_left_row, waist_right_column, waist_right_row, waist_midpoint_column, waist_midpoint_row#, original_rotated_1



def measurements_internal(lx_boundary, ly_boundary,img_size_tuple, location, flag_boundary_direction,  interest_points, unit, measurement_offset):
	"""
	This function determines measurement at key locations like upper hip, lower hip, thigh, knee, backrise in pixels
	
	Inputs:
			lx_boundary 			(numpy array)	:	row co-ordinate of boundary
			ly_boundary 			(numpy array)	:	column co-ordinate of boundary
			img_size_tuple 			(tuple)			:	image size
			location 				(numpy array)	:	location of keypoints
			flag_boundary_direction	(int)			:	boundary direction (clockwise or counter clockwise)
			interest_points 		(numpy array)	:	interest points for measurements
			unit 					(float)			:	inch per pixel
			measurement_offset		(numpy array)	:	distance fro crotch for measurement for given keypoint

	Returns:
			measurements 	(dictionary):	dictionary of measurements


	"""
	rows, columns = img_size_tuple
	bound = np.zeros((rows, columns))
	cv2.drawContours(bound, [lxly2contour(lx_boundary, ly_boundary)], -1, (1), 1)
	
	#print 'inside internal measurements', flag_boundary_direction
	rows_interest_points, columns_interest_points = interest_points.shape
	# for i in range(8):
	# 	cv2.circle(original, (interest_points[i][1], interest_points[i][0]), 10,(255,0,0), -1)

	#show(original, 'ip', imshow_flag)
	crotch_row = interest_points[rows_interest_points-1,0]
	crotch_column = interest_points[rows_interest_points-1,1]
	# cv2.circle(original, (crotch_column, crotch_row), 10, (255,0,0), -1)
	# #print 'crotch row,column', crotch_row, crotch_column



	#unit is inch/pixel and 1/unit is pixel/inch
	upper_hip_row = crotch_row - int(float(measurement_offset['upperhip'])/unit)
	upper_hip_points = np.where(bound[int(upper_hip_row),:]==1)[0].ravel()
	# #print upper_hip_points, int(measurement_offset[0]/unit)
	# for i in range(len(upper_hip_points)):
	# 	cv2.circle(original, (upper_hip_points[i], upper_hip_row),10, (255,0,0), -1)

	upper_hip = 2*(upper_hip_points[1] - upper_hip_points[0])*unit
	# #print 'U Hip', upper_hip

	lower_hip_row = crotch_row - int(float(measurement_offset['lowerhip'])/unit)
	lower_hip_points = np.where(bound[int(lower_hip_row),:]==1)[0].ravel()
	# #print lower_hip_points,  int(measurement_offset[1]/unit)
	# for i in range(len(lower_hip_points)):
	# 	cv2.circle(original, (lower_hip_points[i], lower_hip_row),10, (255,0,0), -1)	

	lower_hip = 2*(lower_hip_points[1] - lower_hip_points[0])*unit
	# #print 'L Hip', lower_hip




#---------------Thigh---------------------------
	offset = abs(int(float(measurement_offset['thigh'])/unit))
	location_crotch = location[7]
	location_thigh_left = -1
	if flag_boundary_direction==1: #counter-clockwise
		location_thigh_left = location_crotch - offset
	else:
		location_thigh_left = location_crotch + offset

	thigh_row_left = int(lx_boundary[int(location_thigh_left)])
	thigh_column_left = int(ly_boundary[int(location_thigh_left)])

	thigh_left_outer_row = thigh_row_left
	thigh_left_outer_column = np.where(bound[int(thigh_row_left),:int(thigh_column_left)]==1)[0][0].ravel()[0]
	# print 'thigh_left_outer_column', thigh_left_outer_column
	# print 'thigh_left_outer_column', thigh_left_outer_column

	thigh_left = 2*(thigh_column_left - thigh_left_outer_column)*unit
	# #print 'thigh left', thigh_row_left, thigh_column_left
	# #print 'thigh left outer', thigh_left_outer_row, thigh_left_outer_column
	# print 'thigh', thigh_column_left,thigh_row_left, thigh_left_outer_column, thigh_left_outer_row
	# cv2.circle(original, (thigh_column_left,thigh_row_left),10, (255,0,0), -1)	
	# cv2.circle(original, (thigh_left_outer_column, thigh_left_outer_row),10, (255,0,0), -1)
	# #show(original, 'thigh left', render = imshow_flag)



	location_thigh_right = -1
	if flag_boundary_direction==1: #counter-clockwise
		location_thigh_right = location_crotch + offset
	else:
		location_thigh_right = location_crotch - offset

	thigh_row_right = lx_boundary[int(location_thigh_right)]
	thigh_column_right = ly_boundary[int(location_thigh_right)]

	thigh_right_outer_row = thigh_row_right
	thigh_right_outer_column = np.where(bound[int(thigh_row_right),int(thigh_column_right+1):]==1)[0][0].ravel()[0] + thigh_column_right+1
	# print 'thigh_right_outer_column', thigh_right_outer_column 
	
	thigh_right = 2*(thigh_right_outer_column - thigh_column_right )*unit
	# #print 'thigh right', thigh_row_right, thigh_column_right
	# #print 'thigh right outer', thigh_right_outer_row, thigh_right_outer_column
	# cv2.circle(original, (int(thigh_column_right), int(thigh_row_right)),10, (255,0,0), -1)		
	# cv2.circle(original, (int(thigh_right_outer_column), int(thigh_right_outer_row)),10, (255,0,0), -1)	
	# #show(original, 'thigh right', render = imshow_flag)


#----------------KNEE---------------------------------
	offset = abs(int(float(measurement_offset['knee'])/unit))
	location_crotch = location[7]
	location_knee_left = -1
	if flag_boundary_direction==1: #counter-clockwise
		location_knee_left = location_crotch - offset
	else:
		location_knee_left = location_crotch + offset

	knee_row_left = lx_boundary[int(location_knee_left)]
	knee_column_left = ly_boundary[int(location_knee_left)]

	knee_left_outer_row = knee_row_left
	knee_left_outer_column = np.where(bound[int(knee_row_left),:int(knee_column_left)]==1)[0][0].ravel()[0]

	knee_left = 2*(knee_column_left - knee_left_outer_column)*unit
	# #print 'knee left', knee_row_left, knee_column_left
	# #print 'knee left outer', knee_left_outer_row, knee_left_outer_column
	# cv2.circle(original, (int(knee_column_left), int(knee_row_left)),10, (255,0,0), -1)		
	# cv2.circle(original, (int(knee_left_outer_column), int(knee_left_outer_row)),10, (255,0,0), -1)	

	location_knee_right = -1
	if flag_boundary_direction==1: #counter-clockwise
		location_knee_right = location_crotch + offset
	else:
		location_knee_right = location_crotch - offset

	knee_row_right = lx_boundary[int(location_knee_right)]
	knee_column_right = ly_boundary[int(location_knee_right)]

	knee_right_outer_row = knee_row_right
	knee_right_outer_column = np.where(bound[int(knee_row_right),int(knee_column_right+1):]==1)[0].ravel()[0] + knee_column_right+1

	knee_right = 2*(knee_right_outer_column - knee_column_right)*unit
	# print 'knee right', knee_row_right, knee_column_right
	# print 'knee right outer', knee_right_outer_row, knee_right_outer_column
	# cv2.circle(original, (int(knee_column_right), int(knee_row_right)),10, (255,0,0), -1)	
	# cv2.circle(original, (int(knee_right_outer_column), int(knee_right_outer_row)),10, (255,0,0), -1)	
	# #show(original, 'knee', render = imshow_flag)





#-------------------BACK_RISE-----------------------
	back_rise_points = np.where(bound[:,int(crotch_column)]==1)[0].ravel()
	# #print 'back rise point', back_rise_points
	back_rise = (crotch_row - back_rise_points[0] ) * unit
	front_rise = back_rise - 1.75

	# cv2.circle(original, (crotch_column, back_rise_points[0] ), 10, (255,0,0), -1)
	# #show(original, 'points', render = imshow_flag)

	thigh = (thigh_right + thigh_left)/2
	knee = (knee_right + knee_left)/2

	upperhip = "{0:.2f}".format(upper_hip)
	lowerhip = "{0:.2f}".format(lower_hip)
	thigh = "{0:.2f}".format(thigh)
	knee = "{0:.2f}".format(knee)
	frontrise = "{0:.2f}".format(front_rise)
	backrise = "{0:.2f}".format(back_rise)

	measurements = {"upperhip": upperhip, "lowerhip": lowerhip, "thigh": thigh, "knee": knee , "frontrise": frontrise, "backrise": backrise }

	return measurements
	



