import math
import random
from collections import defaultdict

import os


cornell_movie_delimiter = '+++$+++' 

directory_location = '../cornell-movie-dialogs-corpus/'

title_metadata_txt_file = 'movie_titles_metadata.txt'

characters_metadata_txt_file = 'movie_characters_metadata.txt'

conversations_txt_file = 'movie_conversations.txt'

movie_lines_txt_file = 'movie_lines.txt'

all_txt_files = [title_metadata_txt_file, characters_metadata_txt_file, conversations_txt_file, movie_lines_txt_file]


def read_txt_file(file_name):
	'''
	Takes in a file name and returns a list of lists where each element of the list corresponds to one line of the text file.
	We know that each important piece of information is separated by the same delimiter, so we use this to create a for loop.
	'''

	useful_information_list = [] #list of lists

	with open(file_name) as f:
		lines = f.readLines()

		for line in lines:
			#print(line)
			split_line = line.split(cornell_movie_delimiter)
			#print(split_line)
			useful_information_list.append(split_line)

	return useful_information_list


def read_text_file_diff_directory(path_name):

	#useful_information_list
	filehandle = open(path_name)
	print(filehandle.read())
	filehandle.close()

def read_movie_text_files(directory_location, text_files):

	text_file_dict = defaultdict(list)
	fileDir = os.path.dirname(os.path.realpath('__file__'))

	for text_file in text_files:

		path = os.path.join(fileDir, directory_location + text_file)
		path = os.path.abspath(os.path.realpath(path))
		print(path)
		text_file_dict[text_file] = read_text_file_diff_directory(path)

	return text_file_dict

read_movie_text_files(directory_location, all_txt_files)




