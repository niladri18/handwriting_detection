import sys
import image as im
import numpy as np
from scipy import optimize
import pdb

#def sigmoid(X,theta):
def sigmoid(z):
	'''
	Calculates the sigmoid function 
	'''
	s = 1./(np.exp(-z)+1)
	return s


def costfunction(theta, *args): 
	'''
	Calculates the Cost function
	'''

	X, y, _lambda = args

	m = X.shape[0] # Number of data
	n = X.shape[1] # Number of features
	
	
	z = np.matmul(X,theta)

	
	h = sigmoid(z)
	# Regularized cost function
	#pdb.set_trace()
	theta2 = np.dot(theta[1:],theta[1:])	
	cost = np.sum( -y*np.log(h) - (1-y)*np.log(1-h) )/m \
		+ _lambda*0.5*theta2/m
	
	#print('cost',cost)	
	return cost


def gradcost(theta, *args):

	'''
	Calculates the gradient of the Cost function
	'''
	
	X, y, _lambda = args
	
	m = X.shape[0] # Number of data
	n = X.shape[1] # Number of features

	z = np.matmul(X,theta)
	
	h = sigmoid(z) # h_theta(x)
	#pdb.set_trace()	

	# unregularized gradient
	grad = np.matmul(h-y,X)/m

	#regularized gradient: leaving first term
	grad[1:] += _lambda*theta[1:]/m 
	
	return grad
	
def oneVsAll(data, _lambda):
	'''
	One vs All Optimization
	'''
	
	X = data.getX()
	m = data.getX().shape[0] # Number of data
	n = data.getX().shape[1] # Number of features
	
	initial_theta = np.zeros(n+1)
	#initial_theta = np.array([-2,-1,1,2])
	
	y = data.gety()
	X = np.concatenate( (np.ones((m,1)), X), axis = 1)
	
	#pdb.set_trace()
	all_theta = []
	k = 10 # Number of classes
	#ans = optimize.fmin_cg(costfunction, initial_theta, fprime = gradcost, args = (X, y, _lambda)   )
	
	for i in range(k):
		t = (y%k == i).astype(int)
		#print(t, i, y)
		#print(np.argmax(t, axis = 0))
		ans = optimize.fmin_cg(costfunction, initial_theta, fprime = gradcost, args = (X, t, _lambda)   )
		all_theta.append(ans)
	
	return np.asarray(all_theta)



	
def predictOneVsAll(data, all_theta):
	X = data.getX()
	m = data.getX().shape[0] # Number of data
	n = data.getX().shape[1] # Number of features
	
	y = data.gety()
	
	#Add ones to X matrix
	X = np.concatenate( (np.ones((m,1)), X), axis = 1)
	
	# Calculates the probability X*(all_theta^T)
	
	#pdb.set_trace()

	z = np.matmul( X, np.transpose(all_theta) )
	
	prob = sigmoid(z)
	#print('prob',prob.shape)
	p = np.argmax(prob, axis = 1) 

	
	accuracy = np.mean( (p == y%10).astype(int) )*100
	
	#print('accuracy',accuracy)
	return accuracy
	
	


'''
def costf(data, _lambda):
	X = data.getX()
	m = data.getX().shape[0] # Number of data
	n = data.getX().shape[1] # Number of features
	
	#theta = np.zeros(n+1)
	theta = np.array([-2,-1,1,2])
	
	y = data.gety()
	X = np.concatenate( (np.ones((m,1)), X), axis = 1)
	z = np.matmul(X,theta)

	
	h = sigmoid(z)
	# Regularized cost function
	
	theta2 = np.zeros(n+1)
	theta2[1:] = theta[1:]*theta[1:]
	
	
	cost = np.sum( -y*np.log(h) - (1-y)*np.log(1-h) )/m \
		+ _lambda*0.5*np.sum(theta2)/m
		
	print('cost',cost)	
	return cost
	
	
def gradf(data, _lambda):
	
	X = data.getX()
	m = data.getX().shape[0] # Number of data
	n = data.getX().shape[1] # Number of features
	
	#theta = np.zeros(n+1)
	theta = np.array([-2,-1,1,2])
	
	y = data.gety()
	X = np.concatenate( (np.ones((m,1)), X), axis = 1)
	z = np.matmul(X,theta)
	h = sigmoid(z)
	pdb.set_trace()
	
	#grad = (h - y).reshape( m,1)*X
	
	grad = np.matmul( (h-y), X )/m
	# unregularized gradient
	#grad = np.sum(grad, axis = 0)/m # sum along rows
	#grad =  np.matmul(h-y,X)
	#print(grad.shape)
	#regularized gradient: leaving first term
	grad[1:] += _lambda*theta[1:]/m 
	
	return grad	
	
'''	

	
	
	
	
	
