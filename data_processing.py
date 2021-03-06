import math
import random
from collections import defaultdict

import os


cornell_movie_delimiter = ' +++$+++ ' 

#directory_location = '../cornell-movie-dialogs-corpus/'

title_metadata_txt_file = 'movie_titles_metadata.txt'

characters_metadata_txt_file = 'movie_characters_metadata.txt'

conversations_txt_file = 'movie_conversations.txt'

movie_lines_txt_file = 'movie_lines.txt'

all_txt_files = [title_metadata_txt_file, characters_metadata_txt_file, conversations_txt_file, movie_lines_txt_file]

all_genres = ['drama', 'horror', 'action', 'comedy', 'crime', 'sci-fi', 'adventure', 'fantasy', 
				'animation', 'romance', 'music', 'war', 'thriller', 'mystery', 'biography', 'western', 
				'musical']


def read_txt_file(file_name):
	'''
	Takes in a file name and returns a list of lists where each element of the list corresponds to one line of the text file.
	We know that each important piece of information is separated by the same delimiter, so we use this to create a for loop.
	'''

	useful_information_list = [] #list of lists

	with open(file_name, encoding = 'latin-1') as f:
		#lines = f.readLines()

		for line in f:
			#print(line)
			#print(line)
			#line.replace(" ", "")
			split_line = line.split(cornell_movie_delimiter)
			#print(split_line)
			#split_line = [x.replace(" ", "") for x in split_line]
			split_line[-1] = split_line[-1][:-1]
			useful_information_list.append(split_line)

	return useful_information_list


# def read_text_file_diff_directory(path_name):

# 	#useful_information_list
# 	filehandle = open(path_name)
# 	print(filehandle.read())
# 	filehandle.close()

def read_movie_text_files(text_files = all_txt_files):

	text_file_dict = defaultdict(list)
	#fileDir = os.path.dirname(os.path.realpath('__file__'))

	for text_file in text_files:

		#path = os.path.join(fileDir, directory_location + text_file)
		#path = os.path.abspath(os.path.realpath(path))
		#print(path)
		text_file_dict[text_file] = read_txt_file(text_file)

	return text_file_dict

# temp_dict = read_movie_text_files(all_txt_files)

# print(temp_dict.keys())

# for key in temp_dict:
# 	print(temp_dict[key][0])

def extract_genres(genre_string):
	list_of_genres = []
	for genre in all_genres:
		if genre in genre_string:
			list_of_genres.append(genre)

	return list_of_genres

def dict_form_title_metadata(info_list):
	#info list is a list of lists like ['m0', '10 things i hate about you', '1999', '6.90', '62847', "['comedy', 'romance']"]
	#returns a dict from movie index number and name to important info

	'''
	returned dictionary has form dict[(movie_index, movie_title)][year/rating/imdb_votes/genres]


	'''
	all_movies_metadata_dict = {}
	movie_index_to_movie_title_dict = {}
	for entry in info_list:
		curr_movie_dict = {
			'year' : entry[2],
			'rating' : entry[3],
			'imdb_votes' : entry[4],
			'genres' : extract_genres(entry[5])
		}
		all_movies_metadata_dict[(entry[0], entry[1])] = curr_movie_dict

		movie_index_to_movie_title_dict[entry[0]] = entry[1]

	return all_movies_metadata_dict, movie_index_to_movie_title_dict


def dict_form_characters_metadata(info_list):
	'''
	info list is like above.
	first returned dictionary is dict[(name, movie index)][gender/credits_ranking]

	second returned dictionary is dict[male/female] = list of male/female characters along with movie index and title
	'''
	all_characters_metadata_dict = {}
	genders_dict = {
		'female': [],
		'male': [],
		'unknown': []
	}

	for entry in info_list:

		curr_char_dict = {
			'gender' : entry[4],
			'ranking' : entry[5]
		}
		name = entry[1]
		movie_index = entry[2]
		#movie_title = entry[3]

		all_characters_metadata_dict[(name, movie_index)] = curr_char_dict

		if entry[4] == 'f':
			genders_dict['female'].append((name, movie_index))
		elif entry[4] == 'm':
			genders_dict['male'].append((name, movie_index))
		else:
			genders_dict['unknown'].append((name, movie_index))

	return all_characters_metadata_dict, genders_dict

def dict_form_movie_lines(info_list):
	'''
	info list is like above
	returns a dictionary with (name, movie_index) to a list of lines spoken

	'''

	char_to_lines_dict = defaultdict(list)

	for entry in info_list:
		name = entry[3]
		movie_index = entry[2]
		line = entry[-1]
		key = (name, movie_index)
		char_to_lines_dict[key].append(line)

	return char_to_lines_dict

# returns list of all lines and all associated genders
def get_labeled_lines(movie_lines_dict, characters_metadata_dict):
	all_lines = []
	labels = []
	for key, lines in movie_lines_dict.items():
		gender = characters_metadata_dict[key]['gender']
		if gender != 'f' and gender != 'm':
			continue
		for line in lines:
			all_lines.append(line.lower())
			labels.append(gender)

	return all_lines, labels

# same as above, but returns equal num of male and female lines
# there are 66228 female lines & 154133 male lines so we take 65000 of each
# this does not take into account different genres or movies
def get_labeled_lines_equal(movie_lines_dict, characters_metadata_dict):
	size = 65000
	male_lines = []
	female_lines = []
	for key, lines in movie_lines_dict.items():
		gender = characters_metadata_dict[key]['gender']
		for line in lines:
			if gender == 'f':
				female_lines.append(line)
			elif gender == 'm':
				male_lines.append(line)
	female_equal = random.sample(female_lines, size)
	male_equal = random.sample(male_lines, size)
	lines = female_equal + male_equal
	genders = ['f' if x < size else 'm' for x in range(size * 2)]
	return lines, genders
	
#similar to the above function except now we want an equal number of male/female 
#lines from each movie instead of just overall.

def get_labeled_lines_equal_per_movie(movie_lines_dict, characters_metadata_dict):
	male_lines = []
	female_lines = []
	movie_line_count = {}
	for key in movie_lines_dict:
		name, movie_index = key
		if movie_index not in movie_line_count:
			movie_line_count[movie_index] = {}
			movie_line_count[movie_index]['counts'] =[0, 0]
			movie_line_count[movie_index]['female_lines'] = []
			movie_line_count[movie_index]['male_lines'] = []
	# movie_line_count = {
	# 	'counts': [0, 0],
	# 	'female_lines': [],
	# 	'male_lines': []
	# } 
	for key, lines in movie_lines_dict.items():
		name, movie_index = key
		gender = characters_metadata_dict[key]['gender']
		for line in lines:
			if len(line.split()) < 10:
				continue
			if gender == 'f':
				movie_line_count[movie_index]['counts'][1] += 1
				movie_line_count[movie_index]['female_lines'].append(line)
			elif gender == 'm':
				movie_line_count[movie_index]['counts'][0] += 1
				movie_line_count[movie_index]['male_lines'].append(line)

	for movie in movie_line_count:
		lower_line_gender_count = min(movie_line_count[movie]['counts'])
		female_equal_lines = random.sample(movie_line_count[movie]['female_lines'], lower_line_gender_count)
		male_equal_lines = random.sample(movie_line_count[movie]['male_lines'], lower_line_gender_count)
		male_lines += male_equal_lines
		female_lines += female_equal_lines

	lines = female_lines + male_lines
	genders = ['f' for x in range(len(female_lines))]
	genders += ['m' for x in range(len(male_lines))]
	return lines, genders





def get_movie_data():
	text_file_dict = read_movie_text_files()
	movie_lines_dict = dict_form_movie_lines(text_file_dict[movie_lines_txt_file])
	characters_metadata_dict, genders_dict = dict_form_characters_metadata(text_file_dict[characters_metadata_txt_file])
	movie_lines_dict = dict_form_movie_lines(text_file_dict[movie_lines_txt_file])

	#getting an equal number of female/male lines from each movie
	return get_labeled_lines_equal_per_movie(movie_lines_dict, characters_metadata_dict)

	#the below line was run for the milestone results
	#return get_labeled_lines_equal(movie_lines_dict, characters_metadata_dict)

def convert_lists_to_dictionaries(text_file_dict):
	movie_metadata_dict, index_to_title_dict = dict_form_title_metadata(text_file_dict[title_metadata_txt_file])

	characters_metadata_dict, genders_dict = dict_form_characters_metadata(text_file_dict[characters_metadata_txt_file])

	movie_lines_dict = dict_form_movie_lines(text_file_dict[movie_lines_txt_file])

	return movie_metadata_dict, index_to_title_dict, characters_metadata_dict, genders_dict, movie_lines_dict

