# EE488B Special Topics in EE <Deep Learning and AlphaGo>
# Fall 2017, School of EE, KAIST

import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt


#-------------------- 
# Parameter Specification
#--------------------
NUM_ITER = 1000      # Number of iterations
nh = 20             # Number of hidden neurons
lr_init = 0.5       # Initial learning rate for gradient descent algorithm
lr_final = 0.1*lr_init     # Final learning rate
#lr_fixed=0.7
var_init = 1      # Standard deviation of initializer
SET_SIZE = 500  #Training dataset size
VAL_SET_SIZE = 50   #Validation dataset size
TEST_SET_SIZE = 100 #Test dataset size

## Symbolic variables
x_ = tf.placeholder(tf.float32, shape=[None,2])
y_ = tf.placeholder(tf.float32, shape=[None,1])

#----------------
## Training dataset generation
#----------------
#x_data = [[0,0], [0,1], [1,0], [1,1]]
#y_data = [[0], [1], [1], [0]]
y_data = np.random.randint(2, size=SET_SIZE) # generate random binary integer array y_data
r = np.random.normal(loc=0.0, scale=1.0, size=SET_SIZE) # generate gaussian random array r
t = np.random.uniform(low=0.0, high=2*np.pi, size=SET_SIZE) # generate uniform random array t
x_data = np.transpose( np.array([r*np.cos(t), r*np.sin(t)]) + np.array(y_data*[5*np.cos(t), 5*np.sin(t)]) ) # generate x_data from r, t, y_data
y_data = y_data.reshape(y_data.shape[0],-1)
#x_data = np.array([[7.383093109163933743e-02, -5.631748944255656614e-02], [-2.349353561378661137e+00,	-6.119387323109627630e+00], [-6.562947972077722625e+00,	-3.770712856152547698e-01], [2.289996564747657004e+00,	-3.713565399668864675e+00], [-1.619053405921073274e+00,	5.643213634373276832e+00]])
#y_data = np.array([[0.000000000000000000e+00], [1.000000000000000000e+00], [1.000000000000000000e+00], [1.000000000000000000e+00], [1.000000000000000000e+00]])

#----------------
## Validation dataset generation
#----------------

## plt data
plt.scatter(x_data[:, 0], x_data[:, 1])
plt.show()

## Weight initialization
# Chosen to be an optimal point to demonstrate convergence to global optimum
#W_init = [[1.0,-1.0],[-1.0,1.0]]
W_init = np.ones(shape=(2, nh), dtype=np.float32)
w_init = np.ones(shape=(nh, 1), dtype=np.float32)
#c_init = [[0.0,0.0]]
c_init = np.zeros(shape=(1, nh), dtype=np.float32)
b_init = np.zeros(shape=(1,1), dtype=np.float32)

#--------------------
# Layer setting
#--------------------
# Weights and biases
W = tf.Variable(W_init+tf.truncated_normal([2, nh], stddev=var_init))
w = tf.Variable(w_init+tf.truncated_normal([nh, 1], stddev=var_init))
c = tf.Variable(c_init+tf.truncated_normal([nh], stddev=var_init))
b = tf.Variable(b_init+tf.truncated_normal([1], stddev=var_init))
#W = tf.Variable(W_init)
#w = tf.Variable(w_init)
#c = tf.Variable(c_init)
#b = tf.Variable(b_init)

#-- Activation setting --
h = tf.nn.relu(tf.matmul(x_, W)+c)
yhat = tf.nn.sigmoid(tf.matmul(h, w)+b)

#-- MSE cost function --
cost = tf.reduce_mean((y_-yhat)**2)
#-- cross entropy cost function -- 
#cost = - tf.reduce_sum(y_*tf.log(yhat))


lr = tf.placeholder(tf.float32, shape=[])
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)

#--------------------
# Run optimization
#--------------------
sess = tf.Session()

sess.run(tf.initialize_all_variables())
for i in range(NUM_ITER):
    #j=np.random.randint(SET_SIZE)    # random index 0~3
    lr_current=lr_init + (lr_final - lr_init) * i / NUM_ITER
#    a=sess.run(train_step, feed_dict={x_:[x_data[j]], y_:[y_data[j]], lr: lr_current})
    a=sess.run(train_step, feed_dict={x_:x_data, y_:y_data, lr: lr_current})
#    a=sess.run(train_step, feed_dict={x_:x_data, y_:y_data, lr: lr_fixed})
    deploy_cost = sess.run(cost, feed_dict={x_:x_data, y_:y_data})
    deploy_yhat = sess.run(yhat, feed_dict={x_:x_data})
    #print('{:2d}: XOR(0,0)={:7.4f}   XOR(0,1)={:7.4f}   XOR(1,0)={:7.4f}   XOR(1,1)={:7.4f}   cost={:.5g}'.\
    #    format(i+1, float(deploy_yhat[0]), float(deploy_yhat[1]), float(deploy_yhat[2]), float(deploy_yhat[3]),float(deploy_cost)))


# define error
error=tf.reduce_mean(tf.abs(y_data-deploy_yhat))

#print("W: ", sess.run(W))
#print("w: ", sess.run(w))
#print("c: ", sess.run(c))
#print("b: ", sess.run(b))
#print("yhat: ", deploy_yhat)
print("training error: ", sess.run(error))

#---------
## Run validation
#---------
# validation dataset generation
y_val_data = np.random.randint(2, size=VAL_SET_SIZE) # generate random binary integer array y_data
r_val = np.random.normal(loc=0.0, scale=1.0, size=VAL_SET_SIZE) # generate gaussian random array r
t_val = np.random.uniform(low=0.0, high=2*np.pi, size=VAL_SET_SIZE) # generate uniform random array t
x_val_data = np.transpose( np.array([r_val*np.cos(t_val), r_val*np.sin(t_val)]) + np.array(y_val_data*[5*np.cos(t_val), 5*np.sin(t_val)]) ) # generate x_data from r, t, y_data
y_val_data = y_val_data.reshape(y_val_data.shape[0],-1)

deploy_val_yhat = sess.run(yhat, feed_dict={x_:x_val_data})
val_error = tf.reduce_mean(tf.abs(y_val_data - deploy_val_yhat))

print("Validation error: ", sess.run(val_error))