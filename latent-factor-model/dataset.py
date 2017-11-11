import numpy as np
from scipy.sparse import coo_matrix


class Dataset:

	def __init__(self):
		self.ind_to_song_map = None
		self.data = None

	def load_triplets(self, path='../data/subset.txt'):
		"""
		Load data triplets and convert to matrix representation
		:param path: path to the dataset
		:return: 
		"""
		data = np.loadtxt(path, delimiter='\t', dtype='str')
		rows, cols, data = data.T

		self.ind_to_song_map = np.unique(cols)
		map_rows = { val:ind for ind,val in enumerate( np.unique(rows) ) }
		map_cols = { val:ind for ind,val in enumerate( np.unique(cols) ) }
		self.data = coo_matrix((data.astype(float), ([map_rows[x] for x in rows], [map_cols[x] for x in cols]))).toarray()
		return self.data