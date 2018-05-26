import math
import operator

from collections import defaultdict


class NaiveBayes:

	def __init__(self, train_dictionary, test_dictionary, stop_words):
		self.train = train_dictionary
		self.test = test_dictionary
		self.stop_words = stop_words

		self.class_counts = defaultdict(float)

		self.word_counts = defaultdict(lambda: defaultdict(float)) # prob(word | class) = dict[class][word]

		self.all_words = set()
		self.total_count = defaultdict(float)


	def filter_stop_words(self, words):
		filtered = []
		for word in words:
	  		if not word in self.stopList and word.strip() != '':
				filtered.append(word)
		return filtered
		#pass

	def classify(self, words):
		'''
		words is a list of strings to classify as male or female

		'''

		female_prob = 0.0
		male_prob = 0.0

		all_lines_count = sum(self.class_counts.values())

		female_prob += math.log(self.class_counts['female']) - math.log(all_lines_count)

		male_prob += math.log(self.class_counts['male']) - math.log(all_lines_count)

		words = self.filter_stop_words(words)

		temp_female = math.log(self.total_count['female']) + len(self.all_words)
		temp_male = math.log(self.total_count['male']) + len(self.all_words)

		for word in words:
			female_prob += math.log(self.wordCounts['female'][word] + 1)
			female_prob -= temp_female

			male_prob += math.log(self.wordCounts['male'][word] + 1)
			male_prob -= temp_male			


		if female_prob >= male_prob:
			return 'female'

		return 'male'

	def add_line_example(self, klass, words):
		'''
		adds a single exxample (movie line) to the model
		klass is the male or female classification
		words is a list of strings
		'''
		self.class_counts[klass] += 1

		self.total_count[klass] += len(words)
		for word in words:
			self.wordCounts[klass][word] += 1

			self.all_words.add(word)

		return


