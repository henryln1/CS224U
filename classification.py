from sklearn import model_selection, naive_bayes, metrics, linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest, f_classif
import pandas
import switchboard_processing
import data_processing
import twitter_processing
import mix_datasets_processing
import blog_processing
import numpy as np 
import operator

np.set_printoptions(threshold=np.nan)

out_file_name = 'features_count_naive_bayes_switchboard_1.txt'


def segmentWords(s):
	"""
	 * Splits lines on whitespace for file reading
	"""
	return s.split()

# Stop words
def stop_read_file(file_name):
	contents = []
	f = open(file_name)
	for line in f:
	  contents.append(line)
	f.close()
	result = segmentWords('\n'.join(contents)) 
	return result
stop_words = stop_read_file('english.stop')

# DATA

# Splitting switchboard
lines, labels = switchboard_processing.get_switchboard_data()
train_x, test_x, train_y, test_y = model_selection.train_test_split(lines, labels)
print("switchboard")

#Splitting movies
# lines, labels = data_processing.get_movie_data()
# train_x, test_x, train_y, test_y = model_selection.train_test_split(lines, labels)

# Training on switchboard, testing on movies
# train_x, train_y = switchboard_processing.get_switchboard_data()
# print("Size of training set: ", len(train_x))
# test_x, test_y = data_processing.get_movie_data()
# print("Size of test set: : ", len(test_x))
# lines = train_x + test_x

# Training on movies, testing on switchboard
# train_x, train_y = data_processing.get_movie_data()
# test_x, test_y = switchboard_processing.get_switchboard_data()
# lines = train_x + test_x

#Splitting tweets
# lines, labels = twitter_processing.get_tweets_data()
# train_x, test_x, train_y, test_y = model_selection.train_test_split(lines, labels)

# splitting blogs
# lines, labels = blog_processing.get_blog_data()
# train_x, test_x, train_y, test_y = model_selection.train_test_split(lines, labels)

#mixing datasets!
# print("Mixing datasets time.")
# lines, labels = mix_datasets_processing.collect_all_datasets()
# train_x, test_x, train_y, test_y = model_selection.train_test_split(lines, labels)

#train on blogs, test on movies
# train_x, train_y = blog_processing.get_blog_data()
# test_x, test_y = data_processing.get_movie_data()
# lines = train_x + test_x


# FEATURES

# Counter Vector
# Unigrams
count_vec = CountVectorizer(analyzer = 'word') # might use stop_words
count_vec.fit(lines)
train_x_count = count_vec.transform(train_x)
test_x_count = count_vec.transform(test_x)


# Bigrams
count_vec_2 = CountVectorizer(analyzer = 'word', ngram_range=(2,2)) # might use stop_words
count_vec_2.fit(lines)
train_x_count_2 = count_vec_2.transform(train_x)
test_x_count_2 = count_vec_2.transform(test_x)

# # TF-IDF
# # Word Level
tfidf_vec_word = TfidfVectorizer(analyzer = 'word')
tfidf_vec_word.fit(lines)
train_x_tfidif_word = tfidf_vec_word.transform(train_x)
test_x_tfidif_word = tfidf_vec_word.transform(test_x)

# # N-gram Level - bigram and trigram
tfidf_vec_ngram= TfidfVectorizer(analyzer = 'word', ngram_range = (2, 3))
tfidf_vec_ngram.fit(lines)
train_x_tfidif_ngram = tfidf_vec_ngram.transform(train_x)
test_x_tfidif_ngram = tfidf_vec_ngram.transform(test_x)

# # possible future features - GloVe, neural nets

all_features = FeatureUnion([("unigrams", count_vec), ("bigrams", count_vec_2), ("tfidif-unigrams", tfidf_vec_word), ("tfidf-ngrams", tfidf_vec_ngram)])
all_features.fit(lines)
train_x_combined = all_features.transform(train_x)
test_x_combined = all_features.transform(test_x)

only_counts = FeatureUnion([("unigrams", count_vec), ("bigrams", count_vec_2)])
only_counts.fit(lines)
train_x_counts = only_counts.transform(train_x)
test_x_counts = only_counts.transform(test_x)

# TRAINING AND TESTING
def train_model(classifier, train_features, label, test_features, print_bool = False, featurizer = None):
	classifier.fit(train_features, label)
	predictions = classifier.predict(test_features)
	# for i in range(len(predictions)):
	#     if predictions[i] != test_y[i]:
	#         print("Utterance: ", test_x[i])
	#         print("Expected: ", test_y[i])
	#         print("Predicted: ", predictions[i])
	if print_bool:
		print(classifier)
		#print(classifier.coef_)
		feature_weights = classifier.feature_count_
		features_1 = feature_weights[0]
		features_2 = feature_weights[1]
		top_features_1 = np.argpartition(features_1, -100)[-100:]
		top_features_2 = np.argpartition(features_2, -100)[-100:]
		top_sorted_1 = top_features_1[np.argsort(features_1[top_features_1])]
		top_sorted_2 = top_features_2[np.argsort(features_2[top_features_2])]
		feature_names_1_sorted = np.asarray(featurizer.get_feature_names())[top_sorted_1]
		feature_names_2_sorted = np.asarray(featurizer.get_feature_names())[top_sorted_1]
		features_1_sorted = features_1[top_sorted_1]
		features_2_sorted = features_2[top_sorted_2]

		features_dict = {}
		for i in range(100):
			name = feature_names_1_sorted[i]
			weight_1 = features_1_sorted[i]
			index_2 = np.where(feature_names_2_sorted == name)
			weight_2 = features_2_sorted[i]
			difference = weight_1 - weight_2
			features_dict[name] = difference

		sorted_feature_differences = sorted(features_dict.items(), key=operator.itemgetter(1))
		print(sorted_feature_differences)

		# only_1_names = np.setdiff1d(feature_names_1_sorted, feature_names_2_sorted)
		# only_2_names = np.setdiff1d(feature_names_2_sorted, feature_names_1_sorted)


		# print("only female features")
		# for name in only_1_names:
		# 	index = np.where(feature_names_1 == name)
		# 	print(name)
		# 	print(features_1_sorted[index])
		
		# print("only male features")
		# for name in only_2_names:
		# 	index = np.where(feature_names_2 == name)
		# 	print(name)
		# 	print(features_2_sorted[index])		


		# print(np.asarray(featurizer.get_feature_names())[top_sorted_1])
		# print(features_1[top_sorted_1])
		# print(np.asarray(featurizer.get_feature_names())[top_sorted_2])
		# print(features_2[top_sorted_2])
		# np.savetxt(out_file_name, feature_weights)

	return metrics.classification_report(predictions, test_y)

print("Naive Bayes")
accuracy = train_model(naive_bayes.MultinomialNB(), train_x_count, train_y, test_x_count, print_bool = True, featurizer = count_vec)
print("Count: ")
print(accuracy)

#print("Printing features of naive bayes to text file...")

accuracy = train_model(naive_bayes.MultinomialNB(), train_x_count_2, train_y, test_x_count_2, print_bool = True, featurizer = count_vec_2)
print("Count (bigrams): ")
print(accuracy)
accuracy = train_model(naive_bayes.MultinomialNB(), train_x_counts, train_y, test_x_counts, print_bool = True, featurizer = only_counts)
print("All Counts: ")
print(accuracy)
accuracy = train_model(naive_bayes.MultinomialNB(), train_x_tfidif_word, train_y, test_x_tfidif_word, print_bool = True, featurizer = tfidf_vec_word)
print("Word TF-IDF: ")
print(accuracy)
accuracy = train_model(naive_bayes.MultinomialNB(), train_x_tfidif_ngram, train_y, test_x_tfidif_ngram, print_bool = True, featurizer = tfidf_vec_ngram)
print("N-Gram TF-IDF: ")
print(accuracy)
accuracy = train_model(naive_bayes.MultinomialNB(), train_x_combined, train_y, test_x_combined, print_bool = True, featurizer = all_features)
print("All: ")
print(accuracy)

# print("=======================================")

# print("Logistic Regression")
# accuracy = train_model(linear_model.LogisticRegression(), train_x_count, train_y, test_x_count)
# print("Count:")
# print(accuracy)
# accuracy = train_model(linear_model.LogisticRegression(), train_x_count_2, train_y, test_x_count_2)
# print("Count (bigrams):")
# print(accuracy)
# accuracy = train_model(linear_model.LogisticRegression(), train_x_counts, train_y, test_x_counts)
# print("All Counts: ")
# print(accuracy)
# accuracy = train_model(linear_model.LogisticRegression(), train_x_tfidif_word, train_y, test_x_tfidif_word)
# print("Word TF-IDF: ")
# print(accuracy)
# accuracy = train_model(linear_model.LogisticRegression(), train_x_tfidif_ngram, train_y, test_x_tfidif_ngram)
# print("N-Gram TF-IDF: ")
# print(accuracy)
# accuracy = train_model(linear_model.LogisticRegression(), train_x_combined, train_y, test_x_combined)
# print("All: ")
# print(accuracy)

# print("=======================================")

# print("SGD")
# accuracy = train_model(linear_model.SGDClassifier(max_iter = 1000), train_x_count, train_y, test_x_count)
# print("Count: ")
# print(accuracy)
# accuracy = train_model(linear_model.SGDClassifier(max_iter = 1000), train_x_count_2, train_y, test_x_count_2)
# print("Count (bigrams) :")
# print(accuracy)
# accuracy = train_model(linear_model.SGDClassifier(max_iter = 1000), train_x_counts, train_y, test_x_counts)
# print("All Counts: ")
# print(accuracy)
# accuracy = train_model(linear_model.SGDClassifier(max_iter = 1000), train_x_tfidif_word, train_y, test_x_tfidif_word)
# print("Word TF-IDF: ")
# print(accuracy)
# accuracy = train_model(linear_model.SGDClassifier(max_iter = 1000), train_x_tfidif_ngram, train_y, test_x_tfidif_ngram)
# print("N-Gram TF-IDF: ")
# print(accuracy)
# accuracy = train_model(linear_model.SGDClassifier(max_iter = 1000), train_x_combined, train_y, test_x_combined)
# print("All: ")
# print(accuracy)

# print("=======================================")

# print("SGD - log loss")
# accuracy = train_model(linear_model.SGDClassifier(loss = 'log', max_iter = 1000), train_x_count, train_y, test_x_count)
# print("Count: ")
# print(accuracy)
# accuracy = train_model(linear_model.SGDClassifier(loss = 'log', max_iter = 1000), train_x_count_2, train_y, test_x_count_2)
# print("Count (bigrams) :")
# print(accuracy)
# accuracy = train_model(linear_model.SGDClassifier(loss = 'log', max_iter = 1000), train_x_counts, train_y, test_x_counts)
# print("All Counts: ")
# print(accuracy)
# accuracy = train_model(linear_model.SGDClassifier(loss = 'log', max_iter = 1000), train_x_tfidif_word, train_y, test_x_tfidif_word)
# print("Word TF-IDF: ")
# print(accuracy)
# accuracy = train_model(linear_model.SGDClassifier(loss = 'log', max_iter = 1000), train_x_tfidif_ngram, train_y, test_x_tfidif_ngram)
# print("N-Gram TF-IDF: ")
# print(accuracy)
# accuracy = train_model(linear_model.SGDClassifier(loss = 'log', max_iter = 1000), train_x_combined, train_y, test_x_combined)
# print("All: ")
# print(accuracy)