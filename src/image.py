import sys
import numpy as np
import matplotlib.pyplot as plt

#####################################################
'''
DEFINE CLASS IMAGE:

This class reads from training set file as pixels 
and stores them as numpy array. 
There is a function 'display' which will display some randomly
selected images after reading from the input file 'input.csv'.
'''
 


class Image:
	def __init__(self):
		self.X = None
		self.y = None
		
	def readfile(self,filename):
		# reads from filename and creates image object
							
		X = [] # to store the data from the image
		y = [] # y value
		while True:
			t = filename.readline().strip().split(',')
			n = len(t)
			if len(t) == 1:
				break
			t = list(map(float, t))	
		
			X.append(t[:-1])
			y.append(t[-1])
		self.X = np.asarray(X, dtype = float)
		self.y = np.asarray(y, dtype = float)	
		
	def getXshape(self):
		if self.X:
			return self.X.shape
		else:
			return 0
	def getYshape(self):
		if self.y:
			return self.y.shape
		else:
			return 0
			
	def getX(self):
		return self.X
	def gety(self):
		return self.y
			
	def display(self, m = 100):
		#This function maps each row to a 20 pixel by 20 pixel 
		
		#Randomly select n points for display, default = 100

		n = self.X.shape[1]
		example_width = round(n**0.5)
		example_height = round(n/example_width)
		
		#Compute number of items to be displayed
		display_rows = round(m**0.5)
		display_cols = round(m/display_rows)	
		

		#between images padding
		pad = 1
		
		#Set up a blank display
		display_array = -np.ones([pad + display_rows*(example_height + pad), \
				pad + display_cols*(example_width + pad)])
		
		# Generate indices of the training matrix x randomly
		r = np.random.randint(0,self.X.shape[0],(m))



		curr_ex = 0

		for i in range(display_rows):
			
			for j in range(display_cols):

				if curr_ex > m:
					break	
				max_val = max(abs(self.X[r[curr_ex],:]))
				#print(r[curr_ex])
				pix = (self.X[r[curr_ex],:]).reshape(example_width,example_height)/max_val
				# Have to flip the array, as images are being read in the opposite direction
				pix = np.flipud(pix)
				idx1 = pad + i*example_height + i
				idx2 = idx1 + example_height 
				idy1 = pad + j*example_width  + j
				idy2 = idy1 +  example_width
				
				
				
				display_array[idy1:idy2, idx1:idx2] = pix
				curr_ex += 1
				
				

		plt.imshow(display_array, aspect = 1)
		plt.show()
		
		return

