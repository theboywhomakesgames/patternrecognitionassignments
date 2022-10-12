import numpy as np
import matplotlib.pyplot as plt

class lc:
	def __init__(self, d):
		print("making a model")
		self.beta = np.random.random((d, )) * 0.0000001

	def fit(self, X, Y, rounds=1000):
		for round in range(rounds):
			# print("optimization round {}".format(round))
			x_shape_0 = X.shape[0]

			P = np.zeros((x_shape_0,))
			W = np.zeros((x_shape_0, x_shape_0))
			for i in range(x_shape_0):
				p0 = self.logistic_func_0(X[i])
				p1 = self.logistic_func_1(X[i])
				W[i][i] = p0 * p1
				P[i] = p0 

			pd1 = np.transpose(X) @ (Y - P)
			pd2 = -np.transpose(X) @ W @ X
			delta = np.linalg.inv(pd2) @ pd1

			self.beta = self.beta - delta

	def predict(self, X):
		p0 = self.logistic_func_0(X)
		p1 = self.logistic_func_1(X)

		return p0 < p1

	def logistic_func_0(self, x):
		bTx = np.transpose(self.beta) @ x
		p = np.power(np.e, bTx)
		return p / (1 + p)

	def logistic_func_1(self, x):
		bTx = np.transpose(self.beta) @ x
		p = np.power(np.e, bTx)
		return 1 / (1 + p)