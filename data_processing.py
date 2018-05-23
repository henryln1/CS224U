import math
import random
from collections import defaultdict


cornell_movie_delimiter = '+++$+++' 

directory_location = '../cornell-movie-dialogs-corpus/'

title_metadata_txt_file = 'movie_titles_metadata.txt'

characters_metadata_txt_file = 'movie_characters_metadata.txt'

conversations_txt_file = 'movie_conversations.txt'

movie_lines_txt_file = 'movie_lines.txt'



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


