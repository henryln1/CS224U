import utils
import random
import os
import tensorflow as tf
from tf_rnn_classifier import TfRNNClassifier
from collections import Counter

glove_dimensionality = 100

home = '..'
base_glove = 'glove.6B'
glove_home = os.path.join(home, base_glove)

glove_text_file = '.' + str(glove_dimensionality) + 'd.txt'

glove_lookup = utils.glove2dict(glove_home + glove_text_file)

def get_vocab(X, n_words=None):
    """Get the vocabulary for an RNN example matrix `X`,
    adding $UNK$ if it isn't already present.

    Parameters
    ----------
    X : list of lists of str
    n_words : int or None
        If this is `int > 0`, keep only the top `n_words` by frequency.

    Returns
    -------
    list of str

    """
    wc = Counter([w for ex in X for w in ex])
    wc = wc.most_common(n_words) if n_words else wc.items()
    vocab = {w for w, c in wc}
    vocab.add("$UNK")
    return sorted(vocab)

class TfBidirectionalRNNClassifier(TfRNNClassifier):
	
	def build_graph(self):
		self._define_embedding()

		self.inputs = tf.placeholder(
			tf.int32, [None, self.max_length])

		self.ex_lengths = tf.placeholder(tf.int32, [None])

		# Outputs as usual:
		self.outputs = tf.placeholder(
			tf.float32, shape=[None, self.output_dim])

		# This converts the inputs to a list of lists of dense vector
		# representations:
		self.feats = tf.nn.embedding_lookup(
			self.embedding, self.inputs)

		# Same cell structure as the base class, but we have
		# forward and backward versions:
		self.cell_fw = tf.nn.rnn_cell.LSTMCell(
			self.hidden_dim, activation=self.hidden_activation)
		
		self.cell_bw = tf.nn.rnn_cell.LSTMCell(
			self.hidden_dim, activation=self.hidden_activation)

		# Run the RNN:
		outputs, finals = tf.nn.bidirectional_dynamic_rnn(
			self.cell_fw,
			self.cell_bw,
			self.feats,
			dtype=tf.float32,
			sequence_length=self.ex_lengths)
	  
		# finals is a pair of `LSTMStateTuple` objects, which are themselves
		# pairs of Tensors (x, y), where y is the output state, according to
		# https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMStateTuple
		# Thus, we want the second member of these pairs:
		last_fw, last_bw = finals          
		last_fw, last_bw = last_fw[1], last_bw[1]
		
		last = tf.concat((last_fw, last_bw), axis=1)
		
		self.feat_dim = self.hidden_dim * 2               

		# Softmax classifier on the final hidden state:
		self.W_hy = self.weight_init(
			self.feat_dim, self.output_dim, 'W_hy')
		self.b_y = self.bias_init(self.output_dim, 'b_y')
		self.model = tf.matmul(last, self.W_hy) + self.b_y  
		


