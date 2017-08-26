
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix as spm

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

	n = A.shape[0]
	I = sp.eye(n)
	res = scalar[0]*A.dot(v)
	yield res
	
	for c in scalar[1:]:
		res = A.dot(c*v + res)
		yield res
	
def expmulit(A,v,scalar=1,k=0):
	for r in __expmulit(A,v,scalar,k): pass	
	return r

def iter_expmulit(A,v,scalar=1,k=0):
	return __expmulit(A,v,scalar,k)

def test_expmulit():

	n = 5
	A = sp.rand(n,n,density=1,format='csr')
	v = np.random.rand(n)	

	u1 = .7*A.dot(v) + \
		 .6*(A.dot(A)).dot(v) + \
		 .5*(A.dot(A).dot(A)).dot(v)
	u2 = expmulit(A,v,[.5,.6,.7])

	print('u2',u2)

	assert (u1 == u2).all(), print(u1,'\n',u2)
	print('Correct')

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

def test_gcn_fw():
	N = 2
	n = 5
	L = [sp.eye(n,format='csr') for i in range(N)]
	X = np.ones((N,n))
	Theta = [.3,.4]

	Y,_ = gcn_fw(L,X,Theta)
	correct = np.ones((N,n))*.7

	assert (Y == correct).all(), print(Y)
	print('Correct')

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

	dX = np.array([expmulit(L[i],dout[i],Theta) for i in range(N)])
	dTheta = [.0]*K

	for i in range(N):
		for k,theta_k in enumerate(iter_expmulit(L[i],X[i],scalar=1,k=K)):
			dTheta[k] += dout[i].dot(theta_k)

	return dX,dTheta

def test_gcn_bw():
	
	N = 2
	n = 5
	L = [sp.eye(n,format='csr') for i in range(N)]
	X = np.ones((N,n))
	Theta = [.3,.4]

	Y,cache = gcn_fw(L,X,Theta)

	dX,dTheta = gcn_bw(Y,cache)
	print(dX)
	print(dTheta)
	
