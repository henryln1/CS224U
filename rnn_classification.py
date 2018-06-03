from word_embeddings_classifier import *
import switchboard_processing
import data_processing
from sklearn.metrics import classification_report
from tf_rnn_classifier import TfRNNClassifier
from sklearn import model_selection, naive_bayes, metrics, linear_model

lines, labels = data_processing.get_movie_data()

train_x, test_x, train_y, test_y = model_selection.train_test_split(lines, labels)


training_vocab = get_vocab(train_x, n_words = 7000)


better_rnn = TfRNNClassifier(
    training_vocab,
    embed_dim=100,
    hidden_dim=50,
    max_length=30,
    hidden_activation=tf.nn.tanh,
    cell_class=tf.nn.rnn_cell.LSTMCell,
    train_embedding=True,
    max_iter=1000,
    eta=0.2) 

_ = better_rnn.fit(train_x, train_y)
better_rnn_dev_predictions = better_rnn.predict(test_x)
print(classification_report(test_y, better_rnn_dev_predictions))