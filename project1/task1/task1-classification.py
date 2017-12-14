## EE488 Fall 2017
## Project 1 Task 1 Classifcation
## Hyoung Uk Shon

import tensorflow as tf
import numpy as np

# Parameter Specification
NUM_ITER = 200      # Number of iterations
nh = 20				# Number of hidden neurons
lr = 5e-3			# learning rate
SET_SIZE = 100		# Size of generated training dataset

# Symbolic variables
x_ = tf.placeholder(tf.float32, shape=[None,2])
y_ = tf.placeholder(tf.float32, shape=[None,1])

# Training data generation
y_data = np.random.randint(2, size=SET_SIZE) # generate random binary integer array y_data
r = np.random.normal(loc=0.0, scale=1.0, size=SET_SIZE) # generate gaussian random array r
t = np.random.uniform(low=0.0, high=2*np.pi, size=SET_SIZE) # generate uniform random array t
x_data = np.transpose( np.array([r*np.cos(t), r*np.sin(t)]) + np.array(y_data*[5*np.cos(t), 5*np.sin(t)]) ) # generate x_data from r, t, y_data

## define the neural network
# Weights and biases
var_init = 0.1      # Standard deviation of initializer
W_init = np.ones(shape=(2,20))
w_init = [[1.0],[1.0]]
c_init = np.ones(shape=(1,20))
b_init = 0.
W = tf.Variable(W_init+tf.truncated_normal([2, nh], stddev=var_init))
w = tf.Variable(w_init+tf.truncated_normal([nh, 1], stddev=var_init))
c = tf.Variable(c_init+tf.truncated_normal([nh], stddev=var_init))
b = tf.Variable(b_init+tf.truncated_normal([1], stddev=var_init))

# Hidden Layer
h = tf.nn.relu(tf.matmul(x_, W)+c) # activated by RELU

# Output layer
y_hat = tf.nn.sigmoid(tf.matmul(h, w)+b) # activated by Sigmoid

# Cost function (cross entropy)
cross_entropy = - tf.reduce_sum(y_*tf.log(y_hat))
# minimize cross entropy
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
# define accuracy to evaluate model
accuracy = tf.reduce_mean((y_-y_hat)**2)

#--------------------
# Run optimization
#--------------------
sess = tf.Session()

sess.run(tf.initialize_all_variables())
for i in range(NUM_ITER):
    a=sess.run(train_step, feed_dict={x_:x_data, y_:y_data})
    sess.run(cross_entropy, feed_dict={x_:x_data, y_:y_data})
    sess.run(y_hat, feed_dict={x_:x_data})

    ## *insert evaluation algorithm here*


    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
    val_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, y_:mnist.validation.labels})
	print("|%d\t|%d\t|%.4f\t|%.4f\t|"%(j+1, i+1, train_accuracy, val_accuracy))


print("W: ", sess.run(W))
print("w: ", sess.run(w))
print("c: ", sess.run(c))
print("b: ", sess.run(b))
