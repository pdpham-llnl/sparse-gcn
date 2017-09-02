import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix as spm

from eval import eval_numerical_gradient,eval_numerical_gradient_array,rel_error

from gcn import *
from layers import softmax_loss
from model_2layer import OneLayer

def assert_diff(x1,x2,tol=1e-16):
	diff = rel_error(x1,x2)
	print('diff',diff)
	assert diff < tol, print(diff)
	print('Correct')

def test_expmulit():

	n = 5
	A = sp.rand(n,n,density=1,format='csr')
	v = np.random.rand(n)	

	u1 = .7*A.dot(v) + \
		 .6*(A.dot(A)).dot(v) + \
		 .5*(A.dot(A).dot(A)).dot(v)
	u2 = expmulit(A,v,[.7,.6,.5])

	diff = rel_error(u1,u2)
	assert  diff < 1e-15, print('Error:',diff)
	print('Correct')

	#Test case 2:
	n = 5
	Theta = [.7,.6,.5]
	L = sp.rand(n,n,density=1,format='csr')
	x = np.random.rand(n)

	expL = sp.eye(n)
	y1 = np.zeros(n)
	for theta in Theta:
		expL = expL.dot(L)
		y1 += theta*expL.dot(x)

	y2 = expmulit(L,x,Theta)

	diff = rel_error(y1,y2)
	assert  diff < 1e-15, print('Error:',diff)

	#Test case 3:
	n = 5
	Theta = [1.0,1.0,1.0]
	L = sp.rand(n,n,density=1,format='csr')
	x = np.random.rand(n)

	expL = sp.eye(n)
	y1 = np.zeros(n)
	for theta in Theta:
		expL = expL.dot(L)
		y1 += theta*expL.dot(x)

	y2 = expmulit(L,x,scalar=1,k=3)

	diff = rel_error(y1,y2)	
	assert  diff < 1e-15, print('Error:',diff)

	#Test case 4:
	n = 5
	A = sp.rand(n,n,density=1,format='csr')
	v = np.random.rand(n)

	u1 = 1.0*A.dot(v) + \
		 1.0*(A.dot(A)).dot(v) + \
		 1.0*(A.dot(A).dot(A)).dot(v)
	u2 = expmulit(A,v,[1.0,1.0,1.0])

	diff = rel_error(u1,u2)
	assert  diff < 1e-15, print('Error:',diff,'\n',u1,'\n',u2)
	print('Correct')

	#Test case 5:
	n = 5
	A = sp.rand(n,n,density=1,format='csr')
	v = np.random.rand(n)

	u1 = [1.0*A.dot(v)] + \
		 [1.0*(A.dot(A)).dot(v)] + \
		 [1.0*(A.dot(A).dot(A)).dot(v)]
	u2 = list(iter_expmulit(A,v,[1.0,1.0,1.0]))

	diff = rel_error(np.array(u1),np.array(u2))
	assert  diff < 1e-15, print('Error:',diff,'\n',u1,'\n',u2)
	print('Correct')


def test_gcn_fw():
	N = 2
	n = 5
	L = [sp.eye(n,format='csr') for i in range(N)]
	X = np.ones((N,n))
	Theta = np.array([.3,.4])

	Y,_ = gcn_fw(L,X,Theta)
	Y_correct = np.ones((N,n))*.7

	diff = rel_error(Y,Y_correct)
	assert  diff < 1e-16, print('Error:',diff)

	#test with random L's
	N = 2
	n = 5
	L = [sp.rand(n,n,density=1,format='csr') for i in range(N)]
	X = np.random.rand(N,n)
	Theta = np.array([.3,.4])

	Y,_ = gcn_fw(L,X,Theta)

	Y_correct = []
	for i in range(N):
		expL = sp.eye(n)
		y = np.zeros(n)
		for theta in Theta:
			expL = expL.dot(L[i])
			y += theta*expL.dot(X[i])
		
		Y_correct.append(y)

	Y_correct = np.array(Y_correct)
	diff = rel_error(Y,Y_correct)
	assert  diff < 1e-15, print('Error:',diff,Y,Y_correct)

	print('Correct!')


def test_gcn_bw():
	
	#Test case 1:
	N = 1
	n = 5
	L = [sp.rand(n,n,density=1,format='csr') for i in range(N)]
	X = np.random.rand(N,n)
	Theta = np.array([.3,.4])

	Y,cache = gcn_fw(L,X,Theta)
	dout = np.random.rand(N,n)
	dX,dTheta = gcn_bw(dout,cache)

	tmp_theta1 = [L[0].dot(X[0]), (L[0].dot(L[0])).dot(X[0])]
	tmp_theta2 = list(iter_expmulit(L[0],X[0],scalar=[1.0,1.0]))

	dTheta_correct1 = [tmp_theta1[0].dot(dout.flatten()),tmp_theta1[1].dot(dout.flatten())]	
	dTheta_correct2 = [tmp_theta2[0].dot(dout.flatten()),tmp_theta2[1].dot(dout.flatten())]

	diff = rel_error(np.array(tmp_theta1),np.array(tmp_theta2))
	assert diff < 1e-16, print(diff,tmp_theta1,tmp_theta2)
	diff = rel_error(np.array(dTheta_correct1), np.array(dTheta_correct2))
	assert diff < 1e-16, print(diff,dTheta_correct1,dTheta_correct2)

	#Test case 2
	N = 1
	n = 5
	L = [sp.rand(n,n,density=1,format='csr') for i in range(N)]
	X = np.random.rand(N,n)
	Theta = np.array([.3,.4])

	Y,cache = gcn_fw(L,X,Theta)
	dout = np.random.rand(N,n)
	dX,dTheta = gcn_bw(dout,cache)

	dX_num = eval_numerical_gradient_array(lambda Z: gcn_fw(L,Z,Theta)[0], X,dout)
	dTheta_num = eval_numerical_gradient_array(lambda Z: gcn_fw(L,X,Z)[0], Theta,dout)
	
	diff = rel_error(dX,dX_num)
	assert diff < 1e-10, print('dX',diff)

	diff = rel_error(dTheta,dTheta_num)
	assert diff < 1e-10, print('dTheta',diff)

	print('Correct!!')

def test_sum_out():

	N=10
	n=5
	label = 3

	print('Test sum_out_fw')
	X = np.random.rand(N,n)
	W = np.random.rand(label)
	Y,_ = sum_out_fw(W,X)
	Y_correct = X.dot(np.ones((n,1))).dot(W.reshape(1,-1))
	assert_diff(Y,Y_correct,tol=1e-15)

	print('Correct')

	Y,cache = sum_out_fw(W,X)
	dout = np.random.rand(N,label)
	dX,dW = sum_out_bw(dout,cache)

	dX_num = eval_numerical_gradient_array(lambda Z:sum_out_fw(W,Z)[0],X,dout)
	dW_num = eval_numerical_gradient_array(lambda Z:sum_out_fw(Z,X)[0],W,dout)

	print(Y.shape,dX.shape,dX_num.shape)
	assert_diff(dX,dX_num,1e-10)
	assert_diff(dW,dW_num,1e-10)


def test_gcn_relu():

	N = 1
	n = 5
	L = [sp.rand(n,n,density=1,format='csr') for i in range(N)]
	X = np.random.rand(N,n)
	Theta = np.array([.3,.4])

	#######################
	print('Test relu_bw')
	_,cache = relu_fw(X)
	dout = np.random.rand(N,n)	
	dX = relu_bw(dout,cache)
	dX_num = eval_numerical_gradient_array(lambda Z:relu_fw(Z)[0],X,dout)
	print('Check relu_bw grad')
	assert_diff(dX,dX_num,tol=1e-10)
	print('Correct')

	#########################
	print('\nTest gcn_relu_fw')

	out,_ = gcn_relu_fw(L,X,Theta)
	out_correct = np.array([expmulit(L[i],X[i],Theta)for i in range(N)])
	out_correct = np.maximum(out_correct,0)
	print('Check gcn_relu_fw output')
	assert_diff(out,out_correct)
	print('Correct')

	#######################

	print('\nTest gcn_relu_bw')
	out_gcn,cache_gcn = gcn_fw(L,X,Theta)
	out_relu,cache_relu = relu_fw(out_gcn)
	
	dout = np.random.rand(N,n)
	dX_relu = relu_bw(dout,cache_relu)
	dX_relu_num = eval_numerical_gradient_array(
		lambda Z:relu_fw(Z)[0],out_gcn,dout)
	print('Check relu grad')
	assert_diff(dX_relu,dX_relu_num,tol=1e-10)

	dX_gcn,dTheta_gcn = gcn_bw(dX_relu,cache_gcn)
	
	dX_gcn_num = eval_numerical_gradient_array(
		lambda Z:gcn_fw(L,Z,Theta)[0],X,dX_relu)
	print('Check gcn X grad')
	assert_diff(dX_gcn,dX_gcn_num,tol=1e-10)
	
	dTheta_gcn_num = eval_numerical_gradient_array(
		lambda Z:gcn_fw(L,X,Z)[0],Theta,dX_relu)
	print('Check gcn Theta grad')
	assert_diff(dTheta_gcn,dTheta_gcn_num,1e-10)

	print('Check gcn relu combined')
	# out_gcn_relu,cache_gcn_relu = gcn_relu_fw(L,X,Theta)
	# dout = np.random.rand(N,n)
	# dX_gcn_relu,dTheta_gcn_relu = gcn_relu_bw(dout,cache_gcn_relu)
	# dX_gcn_relu_num = eval_numerical_gradient_array(
	# 	lambda Z:gcn_relu_fw(L,Z,Theta)[0],X,dout)
	# print('Check gcn_relu dX')
	# assert_diff(dX_gcn_relu,dX_gcn_relu_num)


	_,cache = gcn_relu_fw(L,X,Theta)
	
	dX,dTheta = gcn_relu_bw(dout,cache)

	dX_num = eval_numerical_gradient_array(lambda Z:gcn_relu_fw(L,Z,Theta)[0],X,dout)
	dTheta_num = eval_numerical_gradient_array(lambda Z:gcn_relu_fw(L,X,Z)[0],Theta,dout)
	print('Check grad dX')
	assert_diff(dX,dX_num,1e-10)
	print('Check grad dTheta')
	assert_diff(dTheta,dTheta_num,1e-10)



def test_onelayer_gcn():

	#Test loss func

	N,n,l,K=10,5,3,2

	X = np.random.rand(N,n)
	y = np.random.randint(l,size=N)

	L =	[sp.rand(n,n,density=1,format='csr') for i in range(N)]

	model = OneLayer(N,K,l,weight_scale=1e-3)
	loss,grads =model.loss(X,L,y)
	
	Theta1 = model.params['Theta1']
	W2 = model.params['W2']
	out1 = np.array([expmulit(L[i],X[i],Theta1) for i in range(N)])
	out1 = np.maximum(out1,0)
	out2 = np.dot(out1,np.ones(n)).reshape(-1,1).dot(W2.reshape(1,-1))
	correct_loss,_ = softmax_loss(out2,y)

	print('check loss diff')
	assert_diff(loss,correct_loss)

	#Test gradient
	_,grads = model.loss(X,L,y)

	for name in ['Theta1','W2']:
		grad = grads[name]
		f = lambda _: model.loss(X,L,y)[0]
		grad_num = eval_numerical_gradient(
			f, model.params[name], verbose=False)

		print('Check grad',name)
		assert_diff(grad,grad_num,1e-8)


def sanity_check():
	from solver import Solver
	N,n,l,K=20,10,3,10
	model = OneLayer(N,K,l,reg=.0)
	data = {
		'X_train':np.random.rand(N,n),
		'L_train':[sp.rand(n,n,density=1.0,format='csr') for i in range(N)],
		'y_train':np.random.randint(0,l,size=N),
		'X_val':np.random.rand(N,n),
		'L_val':[sp.rand(n,n,density=1.0,format='csr') for i in range(N)],
		'y_val':np.random.randint(0,l,size=N)
	}

	solver = Solver(model,
					data,
					update_rule = 'sgd_momentum',
					batch_size = 3,
					num_epochs=1000,
					optim_config={
						  'learning_rate': 1e-2,
						},
				   )



