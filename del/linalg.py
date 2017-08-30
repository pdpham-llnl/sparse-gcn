import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix as spm

from dst import DST

def is_dense(A):
	return (type(A) is np.matrix) or (type(A) is np.ndarray)

def dot(A,B):

	if type(A) is spm and type(B) is spm:
		return spm.dot(A,B)
	elif is_dense(A) and is_dense(B):
		return np.dot(A,B)
	elif type(A) is DST and type(B) is DST:
		return A.matmul(B)
	else:
		raise NotImplementedError()
