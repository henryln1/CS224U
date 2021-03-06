from naive_bayes_self import NaiveBayes 
import math
from collections import defaultdict

import re

import itertools

from data_processing import *

import PorterStemmer



porter_stemmer = PorterStemmer.PorterStemmer()

def load_movie_corpus():

	text_file_dict = read_movie_text_files()

	movie_metadata_dict, index_to_title_dict, characters_metadata_dict, genders_dict, movie_lines_dict = convert_lists_to_dictionaries(text_file_dict)

	return_dict = {
		'movie_metadata': movie_metadata_dict,
		'index_title': index_to_title_dict,
		'character_metadata': characters_metadata_dict,
		'genders_character': genders_dict,
		'movie_lines': movie_lines_dict
	}

	return return_dict


def train_test_split(lines):
	#lines = dict(lines)
	#train = {key: value for i, (key, value) in enumerate(lines.viewitems()) if i % 10 != 0}
	#test = {key: value for i, (key, value) in enumerate(lines.viewitems()) if i % 10 == 0}
	train = {}
	test = {}
	keys = list(lines.keys())

	for index in range(len(keys)):
		curr_key = keys[index]
		if index % 10 != 0:
			train[curr_key] = lines[curr_key]
		else:
			test[curr_key] = lines[curr_key]

	return train, test


def segmentWords(s):
	"""
	 * Splits lines on whitespace for file reading
	"""
	return s.split()




def stop_read_file(file_name):
	contents = []
	f = open(file_name)
	for line in f:
	  contents.append(line)
	f.close()
	result = segmentWords('\n'.join(contents)) 
	return result

def cleanup(line):
	#remove nonalphanumeric characters and lowers the string
	#print(line)
	line = re.sub(r'([^\s\w]|_)+', '', line)

	line = line.lower()

	line_list = line.split()

	# for x in range(len(line_list)):
	# 	#curr = line_list[x]
	# 	#if curr[-1] == 'u' and curr[0] == 'u':
	# 	line_list[x] = porter_stemmer.stem(line_list[x])

	return line_list

def train_model(model, all_information_movie_dict, train_dict):

	for key in train_dict:
		character_name, movie_index = key
		character_all_lines = train_dict[key]
		#print(key)

		#line_list_form = cleanup(line)

		gender = all_information_movie_dict['character_metadata'][key]['gender']
		#print(gender)
		for curr_line in character_all_lines:
			line_list_form = cleanup(curr_line)
			model.add_line_example(gender, line_list_form)

	return model

def check_accuracy(model, test_dict, all_information_movie_dict):

	num_correct = 0
	num_total = 0  
	for key in test_dict:
		character_all_lines = test_dict[key]
		#line_list_form = cleanup(line)
		gender = all_information_movie_dict['character_metadata'][key]['gender']
		#print(gender)

		if gender == 'f':
			gender = 'female'
		elif gender == 'm':
			gender = 'male'
		else:
			continue
		#print("Gender: ", gender)
		for curr_line in character_all_lines:
			line_list_form = cleanup(curr_line)
			model_output_gender = model.classify(line_list_form)
			#print("output gender: ", model_output_gender)
			if model_output_gender == gender:
				num_correct += 1
			num_total += 1
	print("Your model achieved accuracy: ", num_correct / num_total)
	return


def main():


	all_information_movie_dict = load_movie_corpus()

	lines_dict = all_information_movie_dict['movie_lines']
	train_dict, test_dict = train_test_split(lines_dict)

	stop_words = set(stop_read_file('english.stop'))
	model = NaiveBayes(train_dict, test_dict, stop_words)

	model = train_model(model, all_information_movie_dict, train_dict)
	check_accuracy(model, test_dict, all_information_movie_dict)
	model.print_word_counts_sorted()
	#model.print_word_counts()
	#model.print_all_words()

if __name__ == "__main__":
	main()
