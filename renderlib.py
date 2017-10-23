# renderlib.py

import matplotlib.pyplot as plt
import cv2


def show(image, title = 'Default', render = True):
	if render == True:
		if len(image.shape)==2:
			plt.imshow(image, cmap = 'Greys')
			plt.title(title)
			plt.show()
		elif len(image.shape)==3:
			plt.imshow(image)
			plt.title(title)
			plt.show()	
		else:
			print 'Invalid Image.'


	return	


def plot(data, title = 'Default', render = True):
	if render == True:
		plt.plot(range(len(data)), data)
		plt.title(title)
		plt.show()
	return



def hist(image, title = None, render = True):
	if render == True:
		if (len(image.shape) == 3):
			b,g,r = cv2.split(image)
			plt.subplot(131)
			plt.hist(b.ravel(),256,[0,256])
			plt.title('b')
			plt.subplot(132)
			plt.hist(g.ravel(),256,[0,256])
			plt.title('g')
			plt.subplot(133)
			plt.hist(r.ravel(),256,[0,256])
			plt.title('r')
			plt.show()
		elif (len(image.shape) == 2):
			plt.hist(image.ravel(),256,[0,256])
			if title is not None:
				plt.title('Histogram' + title)
			plt.show()
		else:
			print 'Invalid Image. Can not compute histogram'
		
	return


def subplot2(images, tile = (-1,-1), title = None, render = False):
	"""
	This function shows multiple images at once using subplot. 
	images is list of images user wants to display. 

	"""

	if render:
		img_len = len(images)
		if len(images[0].shape) == 1:
			# It is 1D data for plot
			tile_x, tile_y = tile
			if tile_x==-1 and tile_y==-1:
				#Decide tile shape
				if img_len<3:
					tile_x = 1
					tile_y = img_len
				elif img_len%2 == 0:
					tile_y = img_len/2
					tile_x = img_len/tile_y
				else:
					tile_y = (img_len+1)/2
					tile_x = (img_len+1)/tile_y
			else:
				tile_x, tile_y = tile

			for i in range(img_len):
				plt.subplot(tile_x, tile_y, i+1)
				plt.plot(range(len(images[i])), images[i])
			plt.show()

		if len(images[0].shape) == 2 or len(images[0].shape) == 3:
			#It is for Images
			tile_x, tile_y = tile
			if tile_x==-1 and tile_y==-1:
				#Decide tile shape
				if img_len<3:
					tile_x = 1
					tile_y = img_len
				elif img_len%2 == 0:
					tile_y = img_len/2
					tile_x = img_len/tile_y
				else:
					tile_y = (img_len+1)/2
					tile_x = (img_len+1)/tile_y
			else:
				tile_x, tile_y = tile

			for i in range(img_len):
				plt.subplot(tile_x, tile_y, i+1)
				if len(images[i].shape)==2:
					plt.imshow(images[i], cmap = 'Greys')
				else:
					plt.imshow(images[i])
				plt.title(str(i))
			plt.show()
	return



def subplot(data, tile = (-1,-1), title = None, render = True):
	"""
	This function shows multiple images at once using subplot. 
	images is list of images user wants to display. 
	"""
	if render:
		data_len = len(data)
		tile_x, tile_y = tile
		if tile_x==-1 and tile_y==-1:
			#Decide tile shape
			if data_len<3:
				tile_x = 1
				tile_y = data_len
			elif data_len%2 == 0:
				tile_y = data_len/2
				tile_x = data_len/tile_y
			else:
				tile_y = (data_len+1)/2
				tile_x = (data_len+1)/tile_y
		else:
			tile_x, tile_y = tile

		for i in range(data_len):
			plt.subplot(tile_x, tile_y, i+1)
			if len(data[i].shape) == 1:
				plt.plot(range(len(data[i])), data[i])
			elif len(data[i].shape) == 2:
				plt.imshow(data[i], cmap = 'Greys')
			elif len(data[i].shape)==3:
				plt.imshow(data[i])
		plt.show()


		return