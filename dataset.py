import numpy as np
from scipy.sparse import coo_matrix


def load_triplets(path='./data/train_triplets.txt'):
	"""
	Load data triplets and convert to matrix representation
	:param path: path to the dataset
	:return: 
	"""
	data = np.loadtxt(path, delimiter='\t', dtype='str')
	rows, cols, data = data.T
	map_rows = { val:ind for ind,val in enumerate( np.unique(rows) ) }
	map_cols = { val:ind for ind,val in enumerate( np.unique(cols) ) }
	return coo_matrix((data.astype(float), ([map_rows[x] for x in rows], [map_cols[x] for x in cols]))).toarray()