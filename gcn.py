import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix as spm

from eval import eval_numerical_gradient_array,rel_error

def __expmulit(A,v,scalar=1,k=0):
	'''
	Recursively calculate exponentials of scalar*A^i*v.
	At each iteration, the result vector is spit out.
	Number of iterations is determined by scalar or k.
	If scalar is iterable (i.e. list) then # of iteration is len(scalar).
	If scalar is a number then # of iteration is determined by k.
	'''
	if isinstance(scalar,(int,float)):
		scalar = [scalar]*k
	
	assert len(scalar) > 0

	scalar = scalar[::-1]

	n = A.shape[0]
	res = scalar[0]*A.dot(v)
	yield res
	
	for c in scalar[1:]:
		res = A.dot(c*v + res)
		yield res
	
def expmulit(A,v,scalar=1,k=0):
	for r in __expmulit(A,v,scalar,k): pass	
	return r

def iter_expmulit(A,v,scalar=1,k=0):
	
	if isinstance(scalar,(int,float)):
		scalar = [scalar]*k
	
	assert len(scalar) > 0

	scalar = scalar[::-1]
	
	exp = v
	for c in scalar:
		exp = A.dot(exp)
		yield c*exp

def gcn_fw(L,X,Theta):

	'''
	X: shape (N,n)
	L: shape (N,n,n)
	Theta: shape(K)
	'''

	assert isinstance(X,(np.ndarray,np.matrix))
	assert isinstance(L,list)
	assert isinstance(L[0],spm)

	N = X.shape[0]
	out = np.array([expmulit(L[i],X[i],Theta) for i in range(N)])
	cache = (L,X,Theta)
	return out, cache

def gcn_bw(dout,cache):
	'''
	X: shape(N,n)
	L: shape(N,n,n)
	Theta: shape(K)

	dout: shape(N,n)
	'''
	L,X,Theta = cache
	N = X.shape[0]
	K = len(Theta)

	dX = np.array([expmulit(L[i].T,dout[i],Theta) for i in range(N)])
	dTheta = [.0]*K

	for i in range(N):
		for k,theta_k in enumerate(iter_expmulit(L[i],X[i],scalar=1.0,k=K)):
			dTheta[k] += dout[i].dot(theta_k)

	return dX,dTheta

def relu_fw(X):
	out = np.maximum(0,X)

	cache = X
	return out,cache

def relu_bw(dout,cache):
	X = cache

	dX = dout * (X >= 0)

	return dX

def gcn_relu_fw(L,X,Theta):

	out_gcn, cache_gcn = gcn_fw(L,X,Theta)
	out_relu, cache_relu = relu_fw(out_gcn)

	cache = (cache_gcn,cache_relu)
	return out_relu, cache

def gcn_relu_bw(dout,cache):

	cache_gcn,cache_relu = cache

	dout_relu = relu_bw(dout,cache_relu)
	dout_gcn = gcn_bw(dout_relu,cache_gcn)

	return dout_gcn

def sum_out_fw(W,X):
	'''
	X: shape(N,n)
	W: shape(n_label,)
	'''

	out = np.dot(np.sum(X,axis=1).reshape(-1,1),
				W.reshape(1,-1))
	cache = W,X

	return out,cache

def sum_out_bw(dout,cache):
	W,X = cache
	dX = np.dot(
			dout.dot(W.reshape(-1,1)),
			np.ones((1,X.shape[1])))	
	dW = np.dot(X.transpose(),dout).sum(axis=0)

	return dX,dW
