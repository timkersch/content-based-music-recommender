import spotipy
import spotipy.oauth2
import spotipy.client
from auth import CLIENT_ID, CLIENT_SECRET
from numpy import argmin

import hdf5_getters as GETTERS

auth = spotipy.oauth2.SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
spotify = spotipy.Spotify(client_credentials_manager=auth)


def levenshtein(s1, s2):
	"""
	Levenstein distance, or edit distance, taken from Wikibooks:
	http://en.wikibooks.org/wiki/Algorithm_implementation/Strings/Levenshtein_distance#Python
	"""
	if len(s1) < len(s2):
		return levenshtein(s2, s1)
	if not s1:
		return len(s2)

	previous_row = xrange(len(s2) + 1)
	for i, c1 in enumerate(s1):
		current_row = [i + 1]
		for j, c2 in enumerate(s2):
			insertions = previous_row[j + 1] + 1
			deletions = current_row[j] + 1
			substitutions = previous_row[j] + (c1 != c2)
			current_row.append(min(insertions, deletions, substitutions))
		previous_row = current_row

	return previous_row[-1]


def get_closest_track(tracklist, target):
	"""
	Find the closest track based on edit distance
	"""
	dists = map(lambda x: levenshtein(x['name'], target), tracklist)
	best = argmin(dists)
	return tracklist[best]


def get_trackid_from_text_search(title, artistname=''):
	"""
	Search for an artist + title using spotify search API
	"""
	return spotify.search(q='artist:' + artistname + ' track:' + title, type='track')['tracks']['items']


def get_preview_from_trackid(trackid):
	try:
		track = spotify.track(trackid)
	except spotipy.client.SpotifyException:
		return ''
	return track['preview_url']


def get_url(h5_file):
	artist_name = GETTERS.get_artist_name(h5_file)
	track_name = GETTERS.get_title(h5_file)
	echo_nest_id = GETTERS.get_track_id(h5_file).lower()

	if echo_nest_id >= 0:
		preview = get_preview_from_trackid(echo_nest_id)
		if preview != '':
			return preview

	res = get_trackid_from_text_search(track_name, artistname=artist_name)
	if len(res) > 0:
		closest_track = get_closest_track(res, track_name)
		preview = get_preview_from_trackid(closest_track['id'])
		return preview

	return None
