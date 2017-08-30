import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix as spm

from linalg import dot

class DST(object):
	"""DST stands for Dense-Sparse Tensor:
		DST can be considered as a tensor dense in 1st dimension, sparse on the rest.
		For now, DST only supports 3-d: 1st d is dense, the other two are sparse.		
	"""	
	def __init__(self, mats):		
		
		assert len(mats) > 0
		shape0 = mats[0].shape
		shape_diff = lambda m: abs(m.shape[0] - shape0[0]) + abs(m.shape[1]-shape0[1])
		assert sum(map(shape_diff,mats)) == 0

		self.mats = mats
		self.shape = (len(mats),*shape0)

	def matmul(self, other):		
		assert self.shape[-2:] == other.shape[-2:] #compatible in the 2 highest dims

		if type(other) is DST and 
			len(self.shape) == len(other.shape):
			mats2 = other.mats
		else:
			

		
		mres = []
		for m1,m2 in zip(self.mats,other.mats):
			mres.append(spm.dot(m1,m2))

		return DST(mres)

	def matmul_bc(self,other):
		assert 	type(other) is spm or 
				type(other) is np.ndarray or
				type(other) is np.matrix
		mres = []


	def test():
		mats1 = [sp.eye(3) for i in range(5)]
		mats2 = [spm(np.arange(9).reshape(3,3)) for i in range(5)]
		dst1 = DST(mats1)
		dst2 = DST(mats2)

		res = dot(dst1,dst2)

		assert np.all(map(lambda mm:mm[0].all()==mm[1].all(), zip(dst1.mats,dst2.mats)))


