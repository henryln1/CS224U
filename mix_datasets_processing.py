import switchboard_processing
import data_processing
import twitter_processing
import random


def zip_and_shuffle(lines, labels, sample_number):
	'''
	this function will take in a list of lines and a list of labels, shuffle them, and then sample from it.
	We want to do this because with how our code is currently set up in returning data and labels, it has all
	the male lines and all the female lines groups together, which isn't good when we want to randomly grab a subset of the data
	'''

	zipped = list(zip(lines, labels))
	random.shuffle(zipped)
	lines, labels = zip(*zipped)

	return lines[:sample_number], labels[:sample_number]





def collect_all_datasets():
	'''
	Will load in the lines/labels for each dataset we want (Twitter, Switchboard, Movies) in this case


	'''
	twitter_lines, twitter_labels = twitter_processing.get_tweets_data()

	movie_lines, movie_labels = data_processing.get_movie_data()

	switchboard_lines, switchboard_labels = switchboard_processing.get_switchboard_data()

	twitter_length = len(twitter_lines)
	movie_length = len(movie_lines)
	switchboard_length = len(switchboard_lines)

	number_drawn_from_each_dataset = min([twitter_length, movie_length, switchboard_length])

	#lines = []
	#labels = []

	twitter_sample_lines, twitter_sample_labels = zip_and_shuffle(twitter_lines, twitter_labels, number_drawn_from_each_dataset)
	movie_sample_lines, movie_sample_labels = zip_and_shuffle(movie_lines, movie_labels, number_drawn_from_each_dataset)
	switchboard_sample_lines, switchboard_sample_labels = zip_and_shuffle(switchboard_lines, switchboard_labels, number_drawn_from_each_dataset)

	lines = twitter_sample_lines + movie_sample_lines + switchboard_sample_lines
	labels = twitter_sample_labels + movie_sample_labels + switchboard_sample_labels

	return lines, labels