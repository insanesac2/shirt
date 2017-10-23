import  cv2,sys, time, os, json
from mylib2 import *
import numpy as np
from renderlib import *

class Pants:
	"""	
	This class implements functions required to perform measurement of pants from single image.


	"""
	def __init__(self, path, specs = {'backrise': '0',  'cuff': '0',  'frontrise': '0',  'inseam': '0', 'knee': '16', 'lowerhip': '2.5', 'outseam': '0', 'thigh': '2', 'upperhip': '6', 'waist': '0'}):
		"""
		initialise parameters to default value. For convenience all the attributes are declared under __init __ to keep track of attributes. 
		"""

		#customer = 0 i.e New customer

		self.error = -1
		self.status = 'success'
		self.measurements = {}

		self.image_resolution_lower_bound = 2560*1920 #5MP
		self.image_resolution_upper_bound = 3264*2448 #8MP
		self.image_resize_factor = 4.78

		if path.endswith('.json'): #returning customer
			self.json_filename = path
			self.specs = specs
			self.measurements = {}
			
			self.__read_as_json()
			self.__measurements()
			return

		else:		
			self.image_path = path
			self.image = cv2.imread(self.image_path)

			self.image_name = (self.image_path).split('.')[0]

			self.json_filename = path.split('.')[0] + '.json'

			# print self.image.shape
			# show(self.image)
		
		if (len(self.image.shape)==2):
			print 'Input image is of Grayscale Type. Can not Proceed'
			self.status = 'Try not using photo-editing applications'
			self.error = 0
			
			return
		else:
			self.rows, self.columns = (self.image).shape[:2]

			if self.rows * self.columns < self.image_resolution_lower_bound:
				self.status = 'Please take the photograph  with camera of 5MP or more'
				self.error = 1
				print 'low resolution image'
				return

			self.__image_resize()
			# show(self.image)

			self.original_for_note_slice = np.copy(self.image)
			self.note_slice = None

			self.original_rotated = None
			self.original_rotated_imwrite = None

			self.image_MSF = None

			self.specs = specs
			self.patch_size_row, self.patch_size_column = 16,12
			


			self.expand_by_length_step = 10

			self.flag_find_note_contour = True

			self.error = -1

			self.mask = None
			self.contours = None
			self.hierarchy = None
			self.pant_contour = None

			self.crotch_row = None
			self.crotch_column = None

			self.cuff_corners_left = None
			self.cuff_corners_right = None

			self.no_of_interest_points = 8
			self.interest_points = None

			self.unit = None

			self.flag_boundary_direction = 1 	#counter-clockwise

			self.measurement_engine()

		return

	def __str__(self):
		if self.error > -1:
			return 'Invalid Instance'
		else:
			return 'image_name: '+ self.image_name + '\n' +'Specs: '+ str(self.specs) + '\n' +'Status: '+ self.status + '\n'

	def __image_resize(self):
		"""
		This function resizes image to a predefined size
		"""
		if self.error >-1:
			return

		if self.image_resolution_lower_bound <=  self.rows*self.columns <= self.image_resolution_upper_bound:
			'''Images within the range are accepted without resizing'''
			# print 'no need of resizing'
			self.rows_small = int(self.rows/self.image_resize_factor)
			self.columns_small = int(self.columns/self.image_resize_factor)
			self.image_small = cv2.resize(self.image, (self.columns_small, self.rows_small))


		elif self.rows * self.columns > self.image_resolution_upper_bound: 
			if self.rows>self.columns:
				row_factor = self.rows/3264.0
				column_factor = self.columns/2448.0
			else:
				row_factor = self.rows/2448.0
				column_factor = self.columns/3264.0

			factor = np.max([row_factor, column_factor]) #To maintain aspect ratio
			# print 'factor', factor

			#resizing image to 8MP resolution
			self.rows = int(self.rows*1.0/factor )
			self.columns = int(self.columns*1.0/factor )
			self.image = cv2.resize(self.image, (self.columns, self.rows))
			# print self.rows, self.columns

			self.rows_small = int(self.rows/self.image_resize_factor)
			self.columns_small = int(self.columns/self.image_resize_factor)
			self.image_small = cv2.resize(self.image, (self.columns_small, self.rows_small))

		return

	def __preprocessing(self):
		"""
			This function corrects orientation of image so as to keep it upright. 
			Then image is blurred and filtered to remove high frequency components.
		"""
		
		if self.error > -1:
			return
		else:
			# correct image orientation
			if self.columns>self.rows:
				self.image = correct_image_orientation(self.image)
				self.rows, self.columns = self.image.shape[:2]

				self.rows_small = int(self.rows/self.image_resize_factor)
				self.columns_small = int(self.columns/self.image_resize_factor)
				self.image_small = cv2.resize(self.image, (self.columns_small, self.rows_small))

				self.original_for_note_slice = np.copy(self.image)

			# show(self.image)
			#Image smoothing
			self.image = cv2.medianBlur(self.image ,5)
			# MSF for smoothing Texture
			self.image_MSF = cv2.pyrMeanShiftFiltering(self.image_small, np.uint((1e-4)*(self.rows_small*self.columns_small))  ,  np.uint((0.5e-4)*(self.rows_small*self.columns_small)) )
		return



	def __pant_localisation(self):
		""" Locate Pant contour """
		if self.error > -1:
			return
		else:
			contrast_image, contrast_image_double, boundary_segments, segments, boundary_segments_AB_values, status = compute_contrast_image(self.image_MSF)

			if status == 'success':

				threshold = cv2.threshold(contrast_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
				# print self.rows, self.columns

				threshold = cv2.resize(threshold, (self.columns, self.rows))

				threshold = cv2.threshold(threshold,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

				threshold_copy = np.copy(threshold)

				try:
					self.contours, self.hierarchy = cv2.findContours(threshold_copy,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

					self.mask = np.zeros((self.rows, self.columns), np.uint8)
					self.pant_contour = sorted(self.contours, key=cv2.contourArea)[-1]
					cv2.drawContours(self.mask, [self.pant_contour], 0, (1), -1) #Draw largest contour
					fourier_descriptor(self.pant_contour, self.rows, self.columns)
				except:
					self.error = 4
					self.status = "Please upload a new Image"
					return
			else:
				self.error = 3
				self.status = 'Server Overloaded'
				return
			return 


	def __note_contour_localisation_from_hierarchy(self):
		"""Localise note contour on pant from hierarchy information"""		
		if self.error > -1:
			return
		else:
			self.flag_find_note_contour = True
			try:
				note_contour = note_contour_from_hierarchy(self.contours, self.hierarchy, self.pant_contour)
			except:
				self.error = 5
				self.status = 'Please place note as per guidlines'
				return
			if note_contour is not None:
				x,y,w,h = cv2.boundingRect(note_contour)
				x,y,w,h = expand_bounding_rect(np.copy(self.mask),x,y,w,h, expand_by_length_step = self.expand_by_length_step)	
				self.note_slice = np.copy(self.original_for_note_slice[y:y+h,x:x+w]) 
				self.original_for_note_slice = None #free memory 
				self.flag_find_note_contour = False
			else:
				self.flag_find_note_contour = True

			return

	def __pant_orientation_correction(self):
		"""
		Correct pant  orientation
		"""
		if self.error > -1:
			return
		else:
			rect  = cv2.minAreaRect(self.pant_contour)
			box = cv2.cv.BoxPoints(rect)
			box = np.int0(box)

			angle = rect[2]
			self.mask, self.original_rotated = correct_object_orientation(np.copy(self.image),angle, np.copy(self.mask))
			self.original_rotated_imwrite = np.copy(self.original_rotated)
			try:
				self.lx_boundary,self.ly_boundary,self.bound = trace_boundary(self.mask.copy()) 
			except:
				self.error = 6
				self.status = 'Please take a photograph as per the guidlines'
			

			return


	def __crotch_localization(self):
		"""
		Locate crotch
		"""
		# print 'inside', np.unique(self.mask)
		# subplot([self.bound, self.mask])
		if self.error > -1:
			return
		else:
			try:
				self.crotch_column, self.crotch_row = locate_crotch(self.bound.copy(), self.mask.copy())
			except:
				self.error = 7
				self.status = 'Please take photograph in unifrom lighting'
				return

			return



	def __crotch_location_based_image_rotation_correction(self):
		if self.error > -1:
			return
		else:
			mid = self.rows/2
			if self.crotch_row > mid:
				"""crotch lies in lower half of the image i.e image is rotated about X-axis. need to correct image rotation"""
				# print 'inside'
				b, g ,r = cv2.split(self.image)
				b,g,r = ((cv2.flip(b,0), cv2.flip(g,0), cv2.flip(r,0)))
				self.image = cv2.merge((cv2.flip(b,1), cv2.flip(g,1), cv2.flip(r,1)))
				# show(res)

				self.mask = cv2.flip( cv2.flip(self.mask, 0) , 1)
				try:
					self.lx_boundary,self.ly_boundary,self.bound = trace_boundary(self.mask.copy())
				except:
					self.error = 6
					self.status = 'Please take a photograph as per the guidlines'
					return
				b, g ,r = cv2.split(self.original_rotated)
				res = np.zeros(self.original_rotated.shape, dtype = np.uint8)
				b,g,r = ((cv2.flip(b,0), cv2.flip(g,0), cv2.flip(r,0)))
				self.original_rotated = cv2.merge((cv2.flip(b,1), cv2.flip(g,1), cv2.flip(r,1)))

				self.original_rotated_imwrite = np.copy(self.original_rotated)
				self.__crotch_localization()
				# subplot([self.image, self.original_rotated, self.original_rotated_imwrite, self.bound, self.mask])
		return


	def __pant_boundary_improvement(self):
		"""
		improve pant boundary by iterating over each boundary pixel and perform thresholding in a predefind window size around each boundary point.
		"""
		if self.error > -1:
			return
		else:

			sel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

			self.bound = cv2.erode(self.bound,sel,iterations = 1)
			try:
				self.lx_boundary, self.ly_boundary, self.bound,self.mask = boundary_to_contour_to_approx_bound(self.lx_boundary, self.ly_boundary, (self.rows, self.columns))
			except:
				self.error = 8
				self.status = 'Please take photograph in unifrom lighting'
				return
			try:
				self.crotch_column, self.crotch_row = locate_crotch(self.bound, self.mask)
			except:
				self.error = 7
				self.status = 'Please take photograph in unifrom lighting'
				return
			try:
				self.lx_boundary, self.ly_boundary, self.bound, self.mask  = improve_boundary(self.lx_boundary, self.ly_boundary, np.copy(self.original_rotated),(self.crotch_row, self.crotch_column), (self.patch_size_row, self.patch_size_column))
			except:
				self.error = 9
				self.status = 'Please take photograph in unifrom lighting'
				return



			if np.all(np.array([self.lx_boundary, self.ly_boundary, self.bound, self.mask]) == -2):
				self.status = 'Please take a photograph with pant in the center'
				print 'Boundary Improvement Failed since pant boundary touches image boundary!!!'
				self.error = 2

			return




	def __locate_cuff_corners(self):
		"""
		locating cuff corners
		"""
		if self.error > -1:
			return
		else:
			try:
				# print np.unique(self.bound)
				# subplot([self.bound, self.mask])
				self.cuff_corners_left = cuff_corner(self.bound[self.crotch_row:-1, :self.crotch_column]*255,(self.bound*255).copy(),  offset_x = 0, offset_y =  self.crotch_row) #Output in OpenCV style
				self.cuff_corners_right = cuff_corner(self.bound[self.crotch_row:-1, self.crotch_column:]*255,(self.bound*255).copy(),  offset_x = self.crotch_column, offset_y = self.crotch_row)
			except:
				self.error = 10
				self.status = 'Please take photograph in unifrom lighting'
				return

			return


	def __improve_above_crotch(self):
		"""
		Improve boundary above crotch by convex hull
		"""
		if self.error > -1:
			return
		else:
			above_crotch = self.bound[0:self.crotch_row,:]
			try:
				self.mask[0:self.crotch_row-1,:] = refine_above_crotch(self.bound[0:self.crotch_row,:])[0:self.crotch_row-1,:]
			except:
				self.error = 11
				self.status = 'Please take photograph in unifrom lighting'
				return
			try:
				self.lx_boundary,self.ly_boundary,self.bound = trace_boundary(self.mask)
			except:
				self.error = 6
				self.status = 'Please take a photograph as per the guidlines'
			return

	def __interest_points_localization(self):
		"""
		Locate interest points for measurements
		"""
		if self.error > -1:
			return
		else:
			above_crotch = self.bound[0:self.crotch_row-50,:]

			# Find waist points
			approximate_waist_length = find_corners(self.lx_boundary,self.ly_boundary, (self.rows, self.columns))

			waist_left_column, waist_left_row, waist_right_column, waist_right_row, waist_midpoint_column, waist_midpoint_row = find_waist_points(above_crotch,self.crotch_row, self.crotch_column, approximate_waist_length,  2, 0, False)

			cuff_left_outer_column, cuff_left_outer_row, cuff_left_inner_column, cuff_left_inner_row  = self.cuff_corners_left[0,0], self.cuff_corners_left[0,1], self.cuff_corners_left[1,0], self.cuff_corners_left[1,1]
			cuff_right_inner_column, cuff_right_inner_row, cuff_right_outer_column, cuff_right_outer_row  = self.cuff_corners_right[0,0], self.cuff_corners_right[0,1], self.cuff_corners_right[1,0], self.cuff_corners_right[1,1]



			#All points in row, column form..since further processing will be in Numpy
			self.no_of_interest_points = 8
			lx = np.zeros(self.no_of_interest_points)
			ly = np.zeros(self.no_of_interest_points)
			lx[0] = waist_left_row
			lx[1] = waist_midpoint_row
			lx[2] = waist_right_row
			lx[3] = cuff_left_outer_row
			lx[4] = cuff_left_inner_row
			lx[5] = cuff_right_inner_row
			lx[6] = cuff_right_outer_row
			lx[7] = self.crotch_row

			ly[0] = waist_left_column
			ly[1] = waist_midpoint_column
			ly[2] = waist_right_column
			ly[3] = cuff_left_outer_column
			ly[4] = cuff_left_inner_column
			ly[5] = cuff_right_inner_column
			ly[6] = cuff_right_outer_column
			ly[7] = self.crotch_column

			#Finding closest points on boundary
			interest_points_x = np.zeros(self.no_of_interest_points)
			interest_points_y = np.zeros(self.no_of_interest_points)
			self.location = np.zeros(self.no_of_interest_points)
			
			try:
				nearest_to_bound[1,self.no_of_interest_points](lx, ly, self.lx_boundary, self.ly_boundary, interest_points_x, interest_points_y, self.location)
			except:
				self.error = 3
				self.status = 'Server Overloaded'
				return
			# print 'Time for nearest points', time.time()-temp, '\n'
			self.interest_points = np.zeros((self.no_of_interest_points,2), dtype = np.int64)
			self.interest_points[:,0] = interest_points_x
			self.interest_points[:,1] = interest_points_y

			# for i in range(self.no_of_interest_points):
			# 	cv2.circle(original_rotated_1, (interest_points[i,1], interest_points[i,0]),20, (255,0,0), -1)


			return


	def __note_detection(self):
		"""
		determine box points 
		"""
		if self.error > -1:
			return
		else:
			try:
				if self.flag_find_note_contour == False:
					note_box_points = find_note( self.note_slice, self.image_name)
				else:
					pass
					#Do processing to locate and measure note

				d1=np.linalg.norm((note_box_points[0] - note_box_points[1]), ord = 2 )
				d2=np.linalg.norm((note_box_points[1] - note_box_points[2]), ord = 2 )
				d3=np.linalg.norm((note_box_points[2] - note_box_points[3]), ord = 2 )
				d4=np.linalg.norm((note_box_points[0] - note_box_points[3]), ord = 2 )
				# print d1, d2, d3, d4

			except Exception as e:
				print e
				self.error = 12
				self.status = 'Please place the note as per the guidlines'
				return

			flag=0
			length1, length2 = (d1+d3)/2, (d2+d4)/2
			
			if length1>length2:
				note_width = length1
				note_height = length2
				flag = 1
			else:
				note_width = length2
				note_height = length1
				flag = 0
			

			# print 'note_height, note_width', note_height, note_width, '\n'
			# print 'Detected note ratio', 2*note_width/note_height, '\n'


			# cv2.imwrite('note_'+ self.image_name, original_rotated_1)

			self.unit = 6.3/note_height #in cm
			# unit = 5.398/note_height
			self.unit = (self.unit)/2.54
			# print self.unit
			return

	def __measurements(self):
		# pixel distance between the interest points
		if self.error > -1:
			return
		else:

			self.flag_boundary_direction = 1 # counterclockwise
			if self.location[4]> self.location[3]	:
				# print 'Boundary trace is anti-clockwise', '\n'
				self.flag_boundary_direction = 1
				waist_length_half = np.linalg.norm((self.interest_points[2] - self.interest_points[0]), ord=2)   # len(lx_boundary) - location[2] - location[0] #L2
				cuff_length_left =   np.linalg.norm((self.interest_points[4] - self.interest_points[3]), ord=2)   # location[4] - location[3] #L2
				cuff_length_right = np.linalg.norm((self.interest_points[6] - self.interest_points[5]), ord=2)  # location[6]- location[5] #L2
				inseam_left = self.location[7] - self.location[4]
				inseam_right = self.location[5] - self.location[7]
				outseam_left = np.linalg.norm((self.interest_points[3] - self.interest_points[0]), ord=2)  # location[3] - location[0] #L2
				outseam_right =  np.linalg.norm((self.interest_points[2] - self.interest_points[6]), ord=2)  #location[2] - location[6] #L2

			else:
				# print 'Measurements are clockwise\n'
				self.flag_boundary_direction = 0


			# print 'waist', waist_length_half
			waist_length = 2*waist_length_half*self.unit
			cuff_left_length = 2*cuff_length_left*self.unit
			cuff_right_length =  2*cuff_length_right*self.unit
			inseam_left_length = inseam_left*self.unit
			inseam_right_length = inseam_right*self.unit
			outseam_left_length = outseam_left*self.unit
			outseam_right_length = outseam_right*self.unit




			#U and L Hip, Thigh, Knee, FR, BR
			# (C-2.5) , (C-1.5), 4, 15, .., ..
			try:
				self.measurements = measurements_internal(self.lx_boundary, self.ly_boundary,(self.rows, self.columns),  self.location, self.flag_boundary_direction, self.interest_points, self.unit,self.specs)  
			except:
				self.error = 13
				self.status = 'Please place the note as per the guidlines'
				return
			self.measurements['waist'] = "{0:.2f}".format(waist_length)
			self.measurements['cuff'] = "{0:.2f}".format((cuff_left_length + cuff_right_length)/2.0)
			self.measurements['inseam'] = "{0:.2f}".format((inseam_left_length + inseam_right_length)/2.0)
			self.measurements['outseam'] = "{0:.2f}".format((outseam_left_length + outseam_right_length)/2.0)

			for key in (self.measurements).keys():
				self.measurements[key] = float(self.measurements[key])

		return

	def __imwrite(self):
		if self.error > -1:
			return
		else:
			draw_color_boundary(self.lx_boundary, self.ly_boundary, self.original_rotated_imwrite)
			for i in range(self.no_of_interest_points):
				cv2.circle(self.original_rotated_imwrite, (self.interest_points[i,1], self.interest_points[i,0]),10, (0,0,255), -1)
			cv2.imwrite(self.image_name +'_res.jpg', self.original_rotated_imwrite)
			return

	def __display(self):
		print self.measurements
		# print self.status, self.error
		return

	def __save_as_json(self):
		if self.error > -1:
			return
		else:
			try:
				data = {}
				data['lx'] = self.lx_boundary.tolist()
				data['ly'] = self.ly_boundary.tolist()
				data['location'] = self.location.tolist()
				data['flag_boundary_direction'] = self.flag_boundary_direction
				data['interest_points'] = self.interest_points.tolist()
				data['unit'] = self.unit
				data['rows'] = self.rows
				data['columns'] = self.columns

				# print self.json_filename
				
				with open(self.json_filename , 'w') as datafile:
					json.dump(data, datafile)
			except Exception as e:
				print e
				self.error = 14
				self.status = "Server undergoing Upgradation. Please try again later."

			return

	def __read_as_json(self):
		if self.error > -1:
			return
		else:
			try:
				with open(self.json_filename, 'r') as datafile:
					data_dict = json.load(datafile)

				self.lx_boundary = np.array(data_dict['lx'])
				self.ly_boundary = np.array(data_dict['ly'])
				self.location = np.array(data_dict['location'])
				self.flag_boundary_direction = data_dict['flag_boundary_direction']
				self.interest_points = np.array(data_dict['interest_points'])
				self.unit = data_dict['unit']
				self.rows = data_dict['rows']
				self.columns = data_dict['columns']
			except Exception as e:
				print e
				self.error = 15
				self.status = "Server undergoing Upgradation. Please try again later."
			return

	def measurement_engine(self):

		start = time.time()

		self.__preprocessing()
					
		self.__pant_localisation()

		self.__note_contour_localisation_from_hierarchy()

		self.__pant_orientation_correction()

		self.__crotch_localization()

		self.__crotch_location_based_image_rotation_correction()

		self.__pant_boundary_improvement()

		self.__locate_cuff_corners()

		self.__improve_above_crotch()

		self.__interest_points_localization()

		self.__note_detection()

		self.__measurements()

		self.__imwrite()

		self.__save_as_json()

		print 'Time: ', time.time() - start
		return


	def matching_engine():
		return






