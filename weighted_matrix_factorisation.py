import numpy as np


class WFS:

	def __init__(self, no_users, no_items, no_factors):
		"""
		:param no_users: Number of users 
		:param no_items: Number of items
		:param no_factors: Number of latent factors to use
		"""
		self.no_users = no_users
		self.no_items = no_items
		self.no_factors = no_factors

		self.X = np.empty(no_users, no_factors)
		self.Y = np.empty(no_items, no_factors)
		self.latent_matrix = np.empty(no_users, no_items)

	def optimise(self, data, alpha, eta, no_iterations):
		"""
		Optimise using ALS
		:param data: the data matrix (users, items)
		:param alpha: scale factor
		:param eta: scale factor
		:param no_iterations: number of ALS iterations to perform
		"""
		P = data > 0.5
		P[P == True] = 1
		P[P == False] = 0
		C = np.ones_like(P) + alpha * np.log(np.ones_like(P) + (np.zeros_like(P) + 1/eta) * data)

		weighted_errors = []
		for i in range(0, no_iterations):
			self._optimise_step()
			error = self._mse(C, P)
			weighted_errors.append(error)
			print('{}th iteration is completed'.format(i))
			print('Error {}'.format(error))
			print('\n')

		self.latent_matrix = np.dot(self.X, self.Y)

	def _optimise_step(self, C, P, reg_term):
		for u, Wu in enumerate(C):
			self.X[u] = np.linalg.solve(np.dot(self.Y, np.dot(np.diag(Wu), self.Y.T)) + reg_term * np.eye(self.no_factors),
			                       np.dot(self.Y, np.dot(np.diag(Wu), P[u].T))).T
		for i, Wi in enumerate(C.T):
			self.Y[:,i] = np.linalg.solve(np.dot(self.X.T, np.dot(np.diag(Wi), self.X)) + reg_term * np.eye(self.no_factors),
			                         np.dot(self.X.T, np.dot(np.diag(Wi), P[:, i])))

	def _mse(self, C, P, X, Y):
		"""
		:param C: confidence matrix 
		:param P: preference matrix
		:param X: User
		:param Y: Item
		:return: The mean squared error
		"""
		return np.sum((C * (P - np.dot(X, Y)))**2)

	def write(self, filename):
		"""
		TODO write data to file
		:param filename: the filename to write to
		"""
		pass