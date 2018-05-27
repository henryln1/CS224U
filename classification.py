from sklearn import model_selection, naive_bayes, metrics, linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas
import switchboard_processing
import data_processing

# DATA

# Splitting switchboard
# lines, labels = switchboard_processing.get_switchboard_data()
# train_x, test_x, train_y, test_y = model_selection.train_test_split(lines, labels)

# Splitting movies
lines, labels = data_processing.get_movie_data()
train_x, test_x, train_y, test_y = model_selection.train_test_split(lines, labels)

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

# FEATURES

# Counter Vector
count_vec = CountVectorizer(analyzer = 'word') # might use stop_words
count_vec.fit(lines)
train_x_count = count_vec.transform(train_x)
test_x_count = count_vec.transform(test_x)

# TF-IDF
# Word Level
tfidf_vec_word = TfidfVectorizer(analyzer = 'word')
tfidf_vec_word.fit(lines)
train_x_tfidif_word = tfidf_vec_word.transform(train_x)
test_x_tfidif_word = tfidf_vec_word.transform(test_x)

# N-gram Level - bigram and trigram
tfidf_vec_ngram= TfidfVectorizer(analyzer = 'word', ngram_range = (2, 3))
tfidf_vec_ngram.fit(lines)
train_x_tfidif_ngram = tfidf_vec_ngram.transform(train_x)
test_x_tfidif_ngram = tfidf_vec_ngram.transform(test_x)

# possible future features - GloVe, neural nets

# TRAINING AND TESTING
def train_model(classifier, train_features, label, test_features):
    classifier.fit(train_features, label)
    predictions = classifier.predict(test_features)    
    return metrics.accuracy_score(predictions, test_y)

# Naive Bayes
print("Naive Bayes")
accuracy = train_model(naive_bayes.MultinomialNB(), train_x_count, train_y, test_x_count)
print("Count: ", accuracy)
accuracy = train_model(naive_bayes.MultinomialNB(), train_x_tfidif_word, train_y, test_x_tfidif_word)
print("Word TF-IDF: ", accuracy)
accuracy = train_model(naive_bayes.MultinomialNB(), train_x_tfidif_ngram, train_y, test_x_tfidif_ngram)
print("N-Gram TF-IDF: ", accuracy)

print("=======================================")

print("Logistic Regression")
accuracy = train_model(linear_model.LogisticRegression(), train_x_count, train_y, test_x_count)
print("Count: ", accuracy)
accuracy = train_model(linear_model.LogisticRegression(), train_x_tfidif_word, train_y, test_x_tfidif_word)
print("Word TF-IDF: ", accuracy)
accuracy = train_model(linear_model.LogisticRegression(), train_x_tfidif_ngram, train_y, test_x_tfidif_ngram)
print("N-Gram TF-IDF: ", accuracy)

print("=======================================")

print("SGD")
accuracy = train_model(linear_model.SGDClassifier(max_iter = 1000), train_x_count, train_y, test_x_count)
print("Count: ", accuracy)
accuracy = train_model(linear_model.SGDClassifier(max_iter = 1000), train_x_tfidif_word, train_y, test_x_tfidif_word)
print("Word TF-IDF: ", accuracy)
accuracy = train_model(linear_model.SGDClassifier(max_iter = 1000), train_x_tfidif_ngram, train_y, test_x_tfidif_ngram)
print("N-Gram TF-IDF: ", accuracy)

print("=======================================")

print("SGD - log loss")
accuracy = train_model(linear_model.SGDClassifier(loss = 'log', max_iter = 1000), train_x_count, train_y, test_x_count)
print("Count: ", accuracy)
accuracy = train_model(linear_model.SGDClassifier(loss = 'log', max_iter = 1000), train_x_tfidif_word, train_y, test_x_tfidif_word)
print("Word TF-IDF: ", accuracy)
accuracy = train_model(linear_model.SGDClassifier(loss = 'log', max_iter = 1000), train_x_tfidif_ngram, train_y, test_x_tfidif_ngram)
print("N-Gram TF-IDF: ", accuracy)