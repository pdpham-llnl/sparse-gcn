import numpy as np
import numpy.linalg as LA
from gcn import *
from optim import sgd_momentum
from layers import softmax_loss

class OneLayer():

	def __init__(self,n_nodes,hidden_dim,n_class,
					weight_scale=1e-3,
					reg = .0):

		self.reg = reg

		np.random.seed(0)		

		self.params = {}
		self.params['Theta1'] = np.random.normal(
                    loc=0,scale=weight_scale,size=hidden_dim)
		self.params['W2'] = np.random.normal(
					loc=0,scale=weight_scale,size=n_class)

	def loss(self,X,L,Y=None):


		out_gcnrelu1,cache_gcnrelu1 = gcn_relu_fw(L,X,self.params['Theta1'])
		out_sum2, cache_sum2 = sum_out_fw(self.params['W2'],out_gcnrelu1)

		scores = out_sum2

		if Y is None:
			return scores

		loss,dout = softmax_loss(scores,Y)

		loss += self.reg * .5 * ( LA.norm(self.params['Theta1'])**2 +
								  LA.norm(self.params['W2'])**2)

		dx2,dw2 = sum_out_bw(dout,cache_sum2)
		dw2 += self.reg * self.params['W2']

		dx1,dtheta1 = gcn_relu_bw(dx2,cache_gcnrelu1)
		dtheta1 += self.reg * self.params['Theta1']

		grads = {'Theta1':dtheta1,'W2':dw2}
		return loss,grads


