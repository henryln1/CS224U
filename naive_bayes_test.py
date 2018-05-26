import sys
import getopt
import os
import math
import operator

from collections import defaultdict

class NaiveBayes:
  class TrainSplit:
	"""Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
	"""
	def __init__(self):
	  self.train = []
	  self.test = []

  class Example:
	"""Represents a document with a label. klass is 'female' or 'male' by convention.
	   words is a list of strings.
	"""
	def __init__(self):
	  self.klass = ''
	  self.words = []


  def __init__(self):
	"""NaiveBayes initialization"""
	self.FILTER_STOP_WORDS = False
	self.BOOLEAN_NB = False
	self.BEST_MODEL = False
	self.stopList = set(self.readFile('data/english.stop'))
	self.numFolds = 10
	# TODO: Add any data structure initialization code here

	self.classCounts = defaultdict(float)
	self.wordCounts = defaultdict(lambda: defaultdict(float)) # prob(word | class) = dict[class][word]
	self.allWords = set()
	self.totalCount = defaultdict(float)

	#binary naive bayes
	self.binaryWordCounts = defaultdict(lambda: defaultdict(float))
	self.binaryTotalCount = defaultdict(float)


  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
  # Boolean (Binarized) features.
  # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
  # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
  # that relies on feature counts.
  #
  # If the BEST_MODEL flag is true, include your new features and/or heuristics that
  # you believe would be best performing on train and test sets. 
  #
  # If any one of the FILTER_STOP_WORDS, BOOLEAN_NB and BEST_MODEL flags is on, the 
  # other two are meant to be off. That said, if you want to include stopword removal
  # or binarization in your best model, write the code accordingl
  # 
  # Hint: Use filterStopWords(words) defined below
  def classify(self, words):
	""" TODO
	  'words' is a list of words to classify. Return 'female' or 'male' classification.
	"""
	# Write code here
	femaleProb = 0.0
	maleProb = 0.0
	allDocumentCount = sum(self.classCounts.values())


	femaleProb += math.log(self.classCounts['female']) - math.log(allDocumentCount) #class probabilities
	maleProb += math.log(self.classCounts['male']) - math.log(allDocumentCount)

	if self.BOOLEAN_NB:
		
		#do boolean naive bayes
		noDuplicates = []
		for i in words:
			if i not in noDuplicates:
				noDuplicates.append(i)
		words = noDuplicates[:]

		tempfemale = math.log(self.binaryTotalCount['female'] + len(self.allWords))
		tempmale = math.log(self.binaryTotalCount['male'] + len(self.allWords))
		for word in words:
			femaleProb += math.log(self.binaryWordCounts['female'][word] + 1)
			femaleProb -= tempfemale

			maleProb += math.log(self.binaryWordCounts['male'][word] + 1)
			maleProb -= tempmale


	elif self.BEST_MODEL:
		noDuplicates = []
		for i in words:
			if i not in noDuplicates:
				noDuplicates.append(i)
		words = noDuplicates[:]
		tempfemale = math.log(self.binaryTotalCount['female'] + 5*len(self.allWords))
		tempmale = math.log(self.binaryTotalCount['male'] + 5*len(self.allWords))

		for word in words:
			femaleProb += math.log(self.binaryWordCounts['female'][word] + 5)
			femaleProb -= tempfemale

			maleProb += math.log(self.binaryWordCounts['male'][word] + 5)
			maleProb -= tempmale

		#personal heuristics

	else:
		if (self.FILTER_STOP_WORDS):
			words = self.filterStopWords(words)
		#regular naive bayes

		tempfemale = math.log(self.totalCount['female'] + len(self.allWords))
		tempmale = math.log(self.totalCount['male'] + len(self.allWords))
		for word in words:

			femaleProb += math.log(self.wordCounts['female'][word] + 1) #one-add smoothing
			femaleProb -= tempfemale

			maleProb += math.log(self.wordCounts['male'][word] + 1) #one-add smoothing
			maleProb -= tempmale #denominator


	if femaleProb >= maleProb:
		return 'female'

	return 'male'
  

  def addExample(self, klass, words):
	"""
	 * TODO
	 * Train your model on an example document with label klass ('female' or 'male') and
	 * words, a list of strings.
	 * You should store whatever data structures you use for your classifier 
	 * in the NaiveBayes class.
	 * Returns nothing
	"""
	# Write code here

	#probability of class is #documents in that class divided by all documents
	#probability of word given class is count of word in that class + 1 divided by number of words in that class + number of different words overall


	self.classCounts[klass] += 1
#	self.totalCount[klass] += len(words)
#	for word in words:
#		self.wordCounts[klass][word] += 1

#		if word not in self.allWords:
#			self.allWords.append(word)


	#binary part
	if self.BOOLEAN_NB or self.BEST_MODEL:
		noDuplicates = []
		for i in words:
			if i not in noDuplicates:
				noDuplicates.append(i)

		self.binaryTotalCount[klass] += len(noDuplicates)

		for word in noDuplicates:
			self.binaryWordCounts[klass][word] += 1
			#if word not in self.allWords:
			self.allWords.add(word)

	else: 
		self.totalCount[klass] += len(words)
		for word in words:
			self.wordCounts[klass][word] += 1
#
			#if word not in self.allWords:
			self.allWords.add(word)



	return
	  

  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
	"""
	 * Code for reading a file.  you probably don't want to modify anything here, 
	 * unless you don't like the way we segment files.
	"""
	contents = []
	f = open(fileName)
	for line in f:
	  contents.append(line)
	f.close()
	result = self.segmentWords('\n'.join(contents)) 
	return result

  
  def segmentWords(self, s):
	"""
	 * Splits lines on whitespace for file reading
	"""
	return s.split()

  
  def trainSplit(self, trainDir):
	"""Takes in a trainDir, returns one TrainSplit with train set."""
	split = self.TrainSplit()
	femaleTrainFileNames = os.listdir('%s/female/' % trainDir)
	maleTrainFileNames = os.listdir('%s/male/' % trainDir)
	for fileName in femaleTrainFileNames:
	  example = self.Example()
	  example.words = self.readFile('%s/female/%s' % (trainDir, fileName))
	  example.klass = 'female'
	  split.train.append(example)
	for fileName in maleTrainFileNames:
	  example = self.Example()
	  example.words = self.readFile('%s/male/%s' % (trainDir, fileName))
	  example.klass = 'male'
	  split.train.append(example)
	return split

  def train(self, split):
	for example in split.train:
	  words = example.words
	  self.addExample(example.klass, words)


  def crossValidationSplits(self, trainDir):
	"""Returns a lsit of TrainSplits corresponding to the cross validation splits."""
	splits = [] 
	femaleTrainFileNames = os.listdir('%s/female/' % trainDir)
	maleTrainFileNames = os.listdir('%s/male/' % trainDir)
	#for fileName in trainFileNames:
	for fold in range(0, self.numFolds):
	  split = self.TrainSplit()
	  for fileName in femaleTrainFileNames:
		example = self.Example()
		example.words = self.readFile('%s/female/%s' % (trainDir, fileName))
		example.klass = 'female'
		if fileName[2] == str(fold):
		  split.test.append(example)
		else:
		  split.train.append(example)
	  for fileName in maleTrainFileNames:
		example = self.Example()
		example.words = self.readFile('%s/male/%s' % (trainDir, fileName))
		example.klass = 'male'
		if fileName[2] == str(fold):
		  split.test.append(example)
		else:
		  split.train.append(example)
	  yield split

  def test(self, split):
	"""Returns a list of labels for split.test."""
	labels = []
	for example in split.test:
	  words = example.words
	  guess = self.classify(words)
	  labels.append(guess)
	return labels
  
  def buildSplits(self, args):
	"""Builds the splits for training/testing"""
	trainData = [] 
	testData = []
	splits = []
	trainDir = args[0]
	if len(args) == 1: 
	  print '[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir)

	  femaleTrainFileNames = os.listdir('%s/female/' % trainDir)
	  maleTrainFileNames = os.listdir('%s/male/' % trainDir)
	  for fold in range(0, self.numFolds):
		split = self.TrainSplit()
		for fileName in femaleTrainFileNames:
		  example = self.Example()
		  example.words = self.readFile('%s/female/%s' % (trainDir, fileName))
		  example.klass = 'female'
		  if fileName[2] == str(fold):
			split.test.append(example)
		  else:
			split.train.append(example)
		for fileName in maleTrainFileNames:
		  example = self.Example()
		  example.words = self.readFile('%s/male/%s' % (trainDir, fileName))
		  example.klass = 'male'
		  if fileName[2] == str(fold):
			split.test.append(example)
		  else:
			split.train.append(example)
		splits.append(split)
	elif len(args) == 2:
	  split = self.TrainSplit()
	  testDir = args[1]
	  print '[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir)
	  femaleTrainFileNames = os.listdir('%s/female/' % trainDir)
	  maleTrainFileNames = os.listdir('%s/male/' % trainDir)
	  for fileName in femaleTrainFileNames:
		example = self.Example()
		example.words = self.readFile('%s/female/%s' % (trainDir, fileName))
		example.klass = 'female'
		split.train.append(example)
	  for fileName in maleTrainFileNames:
		example = self.Example()
		example.words = self.readFile('%s/male/%s' % (trainDir, fileName))
		example.klass = 'male'
		split.train.append(example)

	  femaleTestFileNames = os.listdir('%s/female/' % testDir)
	  maleTestFileNames = os.listdir('%s/male/' % testDir)
	  for fileName in femaleTestFileNames:
		example = self.Example()
		example.words = self.readFile('%s/female/%s' % (testDir, fileName)) 
		example.klass = 'female'
		split.test.append(example)
	  for fileName in maleTestFileNames:
		example = self.Example()
		example.words = self.readFile('%s/male/%s' % (testDir, fileName)) 
		example.klass = 'male'
		split.test.append(example)
	  splits.append(split)
	return splits
  
  def filterStopWords(self, words):
	"""Filters stop words."""
	filtered = []
	for word in words:
	  if not word in self.stopList and word.strip() != '':
		filtered.append(word)
	return filtered

def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL):
  nb = NaiveBayes()
  splits = nb.buildSplits(args)
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
	classifier = NaiveBayes()
	classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
	classifier.BOOLEAN_NB = BOOLEAN_NB
	classifier.BEST_MODEL = BEST_MODEL
	accuracy = 0.0
	for example in split.train:
	  words = example.words
	  classifier.addExample(example.klass, words)
  
	for example in split.test:
	  words = example.words
	  guess = classifier.classify(words)
	  if example.klass == guess:
		accuracy += 1.0

	accuracy = accuracy / len(split.test)
	avgAccuracy += accuracy
	print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) 
	fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy
	
	
def classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, trainDir, testFilePath):
  classifier = NaiveBayes()
  classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
  classifier.BOOLEAN_NB = BOOLEAN_NB
  classifier.BEST_MODEL = BEST_MODEL
  trainSplit = classifier.trainSplit(trainDir)
  classifier.train(trainSplit)
  testFile = classifier.readFile(testFilePath)
  print classifier.classify(testFile)
	
def main():
  FILTER_STOP_WORDS = False
  BOOLEAN_NB = False
  BEST_MODEL = False
  (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
  if ('-f','') in options:
	FILTER_STOP_WORDS = True
  elif ('-b','') in options:
	BOOLEAN_NB = True
  elif ('-m','') in options:
	BEST_MODEL = True
  
  if len(args) == 2 and os.path.isfile(args[1]):
	classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, args[0], args[1])
  else:
	test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL)

if __name__ == "__main__":
	main()
