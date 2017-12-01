import numpy as np
import tensorflow as tf

class DQN: #example - DQN(sess, ni, no, "source Q")
	def __init__(self, sess, input_size, output_size, name="main"):
		self.sess = sess
		self.input_size = input_size
		self.output_size = output_size
		self.net_name = name
		self.build_network()

	def build_network(self, h_size=512, lr=1e-1):
		with tf.variable_scope(self.net_name):
			self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")

			# First layer
			W_h = tf.Variable(tf.truncated_normal(shape=[ni, nh], stddev=0.1))
			b_h = tf.Variable(tf.truncated_normal(shape=[nh], stddev=0.1))
			h = tf.nn.relu(tf.matmul(_X, W_h) + b_h)

			# Output layer
			W_o = tf.Variable(tf.truncated_normal(shape=[nh, no], stddev=0.1))
			b_o = tf.Variable(tf.truncated_normal(shape=[no], stddev=0.1))
			self._Qpred=tf.matmul(h, W_o) + b_o

		# We need to define the parts of the network needed for learning a policy
		self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])

		# Loss function
		self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
		# Learning
		self._train = tf.train.AdamOptimizer(lr).minimize(self._loss)

	def predict(self, state):
		x = np.reshape(state, [1, self.input_size])
		return self.sess.run(self._Qpred, feed_dict={self._X: x})

	def update(self, x_stack, y_stack):
		return self.sess.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})
