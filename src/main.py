import sys
import numpy as np
import matplotlib.pyplot as plt
import image as im
import costFunction as cf
import pdb
from numpy import genfromtxt
#my_data = genfromtxt('my_file.csv', delimiter=',')		
		

		
#####################################################
'''
READ INPUT FILE AND CREATE OBJECT IMAGE IM
'''		
		
f = open('./input/input.csv','r')
data = im.Image()
data.readfile(f)


#print(data.X)
#print(data.y)

#pdb.set_trace()

#####################################################
'''
 Visualize the training data
''' 
data.display()

print('Close to continue')

#####################################################
'''
 Calculate cost function
'''
_lambda = 0.1


#####################################################
'''
SAVE trained THETA in (ans)
'''
theta = cf.oneVsAll(data, _lambda)
np.savetxt("./output/theta.csv", theta, delimiter=",")
print('')
acc = cf.predictOneVsAll(data, theta)
print('Accuracy: ', acc, '%')
