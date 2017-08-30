import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix as spm

from linalg.py import dot as dst_dot
# def transpose(A):
# 	return A.transpose()

def affine_fw(X,W,b):
	#TODO: reshape X for general cases	
    out = np.dot(X,W) + b
    cache = (X,W,b)
    return out,cache

def affine_gcn_fw(X,G,W,b):
	out = dst_add(dst_dot(dst_dot(X,G),W),b)


# def affine_bw(dout, cache):
# 	#dout: upstream derivative shape (N,M): N:rowcount of X
#     X, W, b = cache    
#     ###########################################################################
#     # TODO: Implement the affine backward pass.                               #
#     ###########################################################################
#     dx = dot(dout,)
#     dx = np.reshape(np.dot(dout, np.transpose(w)),x.shape)
#     x_nw_shape = (x.shape[0],np.prod(x.shape[1:]))
#     dw = np.dot(np.transpose(np.reshape(x,x_nw_shape)),dout)
#     db = np.dot(np.ones(dout.shape[0]),dout)
    
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     return dx, dw, db

# def gcn_fw(A,X,W,b):
#     out = A.dot(X).dot(W) + b
#     cache = (A,X,W,b)
#     return out,cache

# def relu_fw(X):
#     return X,spm.maximum(0,X)

# def gcn_relu_fw(A,X,W,b):
#     out_gcn, cache_gcn = gcn_fw(A,X,W,b)
#     out_relu, cache_relu = relu_fw(X)
#     cache = (cache_gcn,cache_relu)
#     return cache,out_relu
    
# def gcn_bw(dout,cache):
    