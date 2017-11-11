import numpy as np


class WFS:

	def __init__(self, no_users, no_items, no_factors=100):
		"""
		:param no_users: Number of users 
		:param no_items: Number of items
		:param no_factors: Number of latent factors to use
		"""
		self.no_users = no_users
		self.no_items = no_items
		self.no_factors = no_factors

		self.X = 5 * np.random.rand(no_users, no_factors)
		self.Y = 5 * np.random.rand(no_factors, no_items)
		self.latent_matrix = np.empty([no_users, no_items])

	def optimise(self, data, alpha=1, eta=0.1, reg_term=0.1, no_iterations=100):
		"""
		Optimise using ALS
		:param data: the data matrix (users, items)
		:param alpha: scale factor
		:param eta: scale factor
		:param no_iterations: number of ALS iterations to perform
		"""
		P = data > 0.5
		C = np.ones_like(P) + alpha * np.log(np.ones_like(P) + (np.zeros_like(P) + 1/eta) * data)

		weighted_errors = []
		for i in range(0, no_iterations):
			self._optimise_step(C, P, reg_term)
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

	def _mse(self, C, P):
		"""
		:param C: confidence matrix 
		:param P: preference matrix
		:param X: User
		:param Y: Item
		:return: The mean squared error
		"""
		return np.sum((C * (P - np.dot(self.X, self.Y)))**2)

	def write(self, filename, ind_song_map=None):
		"""
		:param filename: the filename to write to
		:param ind_song_map: a numpy array from index to song id
		"""
		np.savetxt(filename + '.txt', self.Y.T, fmt='%f')
		if ind_song_map is not None:
			np.savetxt(filename + '-song-map.txt', ind_song_map, fmt='%s')