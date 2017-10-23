#cudalib.py

from numba import jit, autojit, cuda
import os

# setting environment variables


os.environ['NUMBAPRO_NVVM']=r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/nvvm/bin/nvvm64_31_0.dll'

os.environ['NUMBAPRO_LIBDEVICE']=r'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/nvvm/libdevice'





@cuda.jit
def cuda_find_mean_lab(labels,pixel_count_for_labels, lab_image_shifted, cumulative_lab_values):
	"""
		This function finds sum of L, A, B values for a given superpixel label.
		 The cumulative sum is stored in cumulative_lab_values variable.

	Inputs:
			labels (numpy array) : SLIC labesl
			pixel_count_for_labels (numpy array) :
			lab_image_shifted (numpy array) :
			cumulative_lab_values (cumulative ) :


	"""
	x,y = cuda.grid(2)
	row = labels[x,y]
	index = (row,0)
	cuda.atomic.add(pixel_count_for_labels, index,1)


	cuda.atomic.add(cumulative_lab_values,(row,0), lab_image_shifted[x,y,0] ) #CHANNEL 0
	cuda.atomic.add(cumulative_lab_values,(row,1), lab_image_shifted[x,y,1] ) #CHANNEL 1
	cuda.atomic.add(cumulative_lab_values,(row,2), lab_image_shifted[x,y,2] ) #CHANNEL 2

	return


@cuda.jit
def make_contrast_image(labels,contrast_array,  contrast_image_double ):
	"""
	This function maps contrast values from a column array to an image. 
	The column index is same as label value. 

	Inputs: 
			labels (nupy array) : reshaped labels image into a column vector
			contrast_array (numpy array) : column vector of same size of labels
			contrast_image_double (numpy array) : column vector of same size of labels

	"""
	x,y = cuda.grid(2)
	row = labels[x,y]

	cuda.atomic.add(contrast_image_double, (x,y), contrast_array[0,row]) #CHANEEL 0
	# cuda.atomic.add(contrast_image_double, (x,y), contrast_array[row,1]) #CHANEEL 0
	# cuda.atomic.add(contrast_image_double, (x,y), contrast_array[row,2]) #CHANEEL 0

	return

@cuda.jit
def nearest_to_bound(lx, ly, lx_boundary, ly_boundary,  nearest_array_x, nearest_array_y, location):

	"""
	This function finds out a point on boundary which is nearest to a query point. 
	It is achieved by calculating L2 norm for the query point with all the boundary points.

	Inputs:
			lx (numpy array) : y co-ordinate of interest point
			ly (numpy array) : x co-ordinate of interest point
			lx_boundary (numpy array) : y co-ordinate of boundary points
			ly_boundary (numpy array) : x co-ordinate of boundary points
			nearest_array_x (numpy array) : nearest point on boundary to query point
			nearest_array_y (numpy array) : nearest point on boundary to query point
			location (numpy array) : location of nearest point on boundary in boundary array
	"""
	index = cuda.threadIdx.x
	# #compute diff with all elements
	min_distance = 1e10 #initialise to some value higher than 0
	for i in range(len(lx_boundary)):
		dist = (lx_boundary[i]-lx[index])*(lx_boundary[i]-lx[index]) + (ly_boundary[i]-ly[index])*(ly_boundary[i]-ly[index])
		if (dist)<min_distance:
			min_distance = dist
			nearest_array_x[index] = lx_boundary[i]
			nearest_array_y[index] = ly_boundary[i]
			location[index] = i

	return

