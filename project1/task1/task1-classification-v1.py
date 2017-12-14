# EE488B Special Topics in EE <Deep Learning and AlphaGo>
# Fall 2017, School of EE, KAIST

import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

## dataset generator
def generate_dataset(SET_SIZE, seed=None):
    np.random.seed(seed)
    y_data = np.random.randint(2, size=SET_SIZE) # generate binary integer array y_data
    np.random.seed(seed)
    r = np.random.normal(loc=0.0, scale=1.0, size=SET_SIZE) # generate gaussian random array r
    np.random.seed(seed)
    t = np.random.uniform(low=0.0, high=2*np.pi, size=SET_SIZE) # generate uniform random array t
    x_data = np.transpose( np.array([r*np.cos(t), r*np.sin(t)]) + np.array(y_data*[5*np.cos(t), 5*np.sin(t)]) ) # calculate x_data
    y_data = y_data.reshape(y_data.shape[0],-1)
    return (x_data, y_data)

## dataset evaluation. calculates accuracy from the labels and unrectified yhat.
def accuracy(y, yhat):
    if(np.size(y) != np.size(yhat)):    # error exception
        print("error")
        return None
    yhat_rectified = np.piecewise(yhat, [yhat < 0.5, yhat >= 0.5], [0, 1])
    eval_array = np.logical_not(np.logical_xor(y, yhat_rectified))
    score = np.sum(eval_array)
    return 100.0*score/eval_array.size

#-------------------- 
# Parameter Specification
#--------------------
NUM_ITER = 2000      # Number of iterations
nh = 20             # Number of hidden neurons
lr_init = 0.5       # Initial learning rate for gradient descent algorithm
lr_final = 0.1*lr_init     # Final learning rate
#lr_fixed=0.4
var_init = 0.1      # Standard deviation of initializer
SET_SIZE = 800  #Training dataset size
VAL_SET_SIZE = 400   #Validation dataset size
TEST_SET_SIZE = 200 #Test dataset size


#----------------
## Training dataset generation
#----------------
(x_data, y_data) = generate_dataset(SET_SIZE)

## Symbolic variables
x_ = tf.placeholder(tf.float32, shape=[None,2])
y_ = tf.placeholder(tf.float32, shape=[None,1])

## Weight initialization
# Chosen to be an optimal point to demonstrate convergence to global optimum
W_init = np.zeros(shape=(2, nh), dtype=np.float32)
w_init = np.zeros(shape=(nh, 1), dtype=np.float32)
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

#-- Activation setting --
h = tf.nn.relu(tf.matmul(x_, W)+c)
yhat = tf.nn.sigmoid(tf.matmul(h, w)+b)

#-- MSE cost function --
cost = tf.reduce_mean((y_-yhat)**2)
#-- cross entropy cost function -- 
#cost = - tf.reduce_sum(y_*tf.log(yhat))

## gradient descent optimizer
lr = tf.placeholder(tf.float32, shape=[])
train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

#--------------------
# Run optimization
#--------------------
sess = tf.Session()

sess.run(tf.initialize_all_variables())
for i in range(NUM_ITER):
    ## Running Optimizer
    ## fixed learning_rate
#    a=sess.run(train_step, feed_dict={x_:x_data, y_:y_data, lr: lr_fixed})
    ## decremental learning_rate
    lr_current=lr_init + (lr_final - lr_init) * i / NUM_ITER
    a=sess.run(train_step, feed_dict={x_:x_data, y_:y_data, lr: lr_current})
    ## randomly select training point
#    j=np.random.randint(SET_SIZE)    # random index 0~3
#    a=sess.run(train_step, feed_dict={x_:[x_data[j]], y_:[y_data[j]], lr: lr_current})
    if((i+1)%100==0):
        deploy_cost = sess.run(cost, feed_dict={x_:x_data, y_:y_data})
        print('Iter: ', i+1, 'Cost: ', deploy_cost)

        
## print training accuracy
deploy_yhat = sess.run(yhat, feed_dict={x_:x_data})
train_accuracy = accuracy(y_data, deploy_yhat)
print("Train accuracy: ", train_accuracy, "%", "SIZE=", SET_SIZE)

#---------
## Run validation
#---------
# validation dataset generation
(x_val_data, y_val_data) = generate_dataset(VAL_SET_SIZE)
deploy_val_yhat = sess.run(yhat, feed_dict={x_:x_val_data})
val_accuracy = accuracy(y_val_data, deploy_val_yhat)
print("Validation accuracy: ", val_accuracy, "%", "SIZE=", VAL_SET_SIZE)

#---------
## Run test
#---------
# validation dataset generation
(x_test_data, y_test_data) = generate_dataset(TEST_SET_SIZE)
deploy_test_yhat = sess.run(yhat, feed_dict={x_:x_test_data})
test_accuracy = accuracy(y_test_data, deploy_test_yhat)
print("Test accuracy: ", test_accuracy, "%", "SIZE=", TEST_SET_SIZE)

# save trained parameters
W_model = np.array(sess.run(W))
w_model = np.array(sess.run(w))
c_model = np.array(sess.run(c))
b_model = np.array(sess.run(b))
np.savetxt("W_model.csv", W_model, delimiter=",")
np.savetxt("w_small_model.csv", w_model, delimiter=",")
np.savetxt("c_model.csv", c_model, delimiter=",")
np.savetxt("b_model.csv", b_model, delimiter=",")


# plot test data
plt.figure(1, figsize=(10,10))
plt.xlim(-8,8)
plt.ylim((-8,8))
plt.scatter(x_test_data[:, 0], x_test_data[:, 1])
plt.title('Test Dataset', loc='left')

# plot criterion lines
for i in range(nh):
    x_axis = np.array(range(-8, 9, 1))
    plt.plot(x_axis, -(W_model[0][i]*x_axis + c_model[0][i])/W_model[1][i], 'r')
plt.show()
