import os
import glob
import dependencies.hdf5_getters as GETTERS


class Dataset:

	def __init__(self):
		self.msd_subset_path = './data/MillionSongSubset'
		self.msd_subset_data_path = os.path.join(self.msd_subset_path, 'data')
		self.msd_subset_addf_path = os.path.join(self.msd_subset_path, 'AdditionalFiles')
		self.ext = '.h5'
		assert os.path.isdir(self.msd_subset_path), 'wrong path'

	def songs_generator(self):
		"""
		Generator function yielding every song
		"""
		for root, dirs, files in os.walk(self.msd_subset_data_path):
			files = glob.glob(os.path.join(root,'*' + self.ext))
			for f in files:
				h5 = GETTERS.open_h5_file_read(f)
				yield h5
				h5.close()

	def apply_to_all_files(self, basedir, func=lambda x: x, ext='.h5'):
		"""
		From a base directory, go through all subdirectories,
		find all files with the given extension, apply the
		given function 'func' to all of them.
		If no 'func' is passed, we do nothing except counting.
		INPUT
		   basedir  - base directory of the dataset
		   func     - function to apply to all filenames
		   ext      - extension, .h5 by default
		RETURN
		   number of files
		"""
		cnt = 0
		# iterate over all files in all subdirectories
		for root, dirs, files in os.walk(basedir):
			files = glob.glob(os.path.join(root,'*'+ext))
			# count files
			cnt += len(files)
			# apply function to all files
			for f in files :
				func(f)
		return cnt