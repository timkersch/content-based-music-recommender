from dataset import Dataset
from weighted_matrix_factorisation import WFS


def run():
	dataset = Dataset()
	data = dataset.load_triplets()
	users, songs = data.shape
	wfs = WFS(users, songs)
	wfs.optimise(data)
	wfs.write('../data/latent/subset-latent-factors', dataset.ind_to_song_map)

if __name__ == '__main__':
	run()