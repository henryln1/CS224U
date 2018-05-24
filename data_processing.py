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

def read_movie_text_files(text_files):

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
	for entry in info_list:
		curr_movie_dict = {
			'year' : entry[2],
			'rating' : entry[3],
			'imdb_votes' : entry[4],
			'genres' : extract_genres(entry[5])
		}
		all_movies_metadata_dict[(entry[0], entry[1])] = curr_movie_dict

	return all_movies_metadata_dict


def convert_lists_to_dictionaries(text_file_dict):
	movie_metadata_dict = dict_form_title_metadata(text_file_dict[title_metadata_txt_file])



