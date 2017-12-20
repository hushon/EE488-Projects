import tensorflow as tf
import numpy as np
from boardgame import game1, game2, game3, game4, data_augmentation

# Choose game tic-tac-toe
game = game2()

#####################################################################
"""                    DEFINE HYPERPARAMETERS                     """
#####################################################################
# Initial Learning Rate
alpha = 0.001
# size of minibatch
size_minibatch = 1024
# training epoch
max_epoch = 10
# number of training steps for each generation
n_train_list = [10000, 50000]
n_test_list = [1000, 1000]

####################################################################
"""                COMPUTATIONAL GRAPH CONSTRUCTION               """
#####################################################################

### DEFINE OPTIMIZER ###
def network_optimizer(Y, Y_, alpha, scope):
    # Cross entropy loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Y, labels = Y_))
    # Parameters in this scope
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope)
    # L2 regularization
    for i in range(len(variables)):
        loss += 0.0001 * tf.nn.l2_loss(variables[i])
    # Optimizer
    optimizer = tf.train.AdamOptimizer(alpha).minimize(loss,\
            var_list = variables)
    return loss, optimizer


### NETWORK ARCHITECTURE ###
def network(state, nx, ny):
    # Set variable initializers
    init_weight = tf.random_normal_initializer(stddev = 0.1)
    init_bias = tf.constant_initializer(0.1)

    # Create variables "weights1" and "biases1".
    weights1 = tf.get_variable("weights1", [3, 3, 3, 30], initializer = init_weight)
    biases1 = tf.get_variable("biases1", [30], initializer = init_bias)

    # Create 1st layer
    conv1 = tf.nn.conv2d(state, weights1, strides = [1, 1, 1, 1], padding = 'SAME')
    out1 = tf.nn.relu(conv1 + biases1)

    # Create variables "weights2" and "biases2".
    weights2 = tf.get_variable("weights2", [3, 3, 30, 50], initializer = init_weight)
    biases2 = tf.get_variable("biases2", [50], initializer = init_bias)

    # Create 2nd layer
    conv2 = tf.nn.conv2d(out1, weights2, strides = [1, 1, 1, 1], padding ='SAME')
    out2 = tf.nn.relu(conv2 + biases2)
   
    # Create variables "weights1fc" and "biases1fc".
    weights1fc = tf.get_variable("weights1fc", [nx * ny * 50, 100], initializer = init_weight)
    biases1fc = tf.get_variable("biases1fc", [100], initializer = init_bias)
    
    # Create 1st fully connected layer
    fc1 = tf.reshape(out2, [-1, nx * ny * 50])
    out1fc = tf.nn.relu(tf.matmul(fc1, weights1fc) + biases1fc)

    # Create variables "weights2fc" and "biases2fc".
    weights2fc = tf.get_variable("weights2fc", [100, 3], initializer = init_weight)
    biases2fc = tf.get_variable("biases2fc", [3], initializer = init_bias)

    # Create 2nd fully connected layer
    return tf.matmul(out1fc, weights2fc) + biases2fc


# Input
S = tf.placeholder(tf.float32, shape = [None, game.nx, game.ny, 3], name = "S")

# Define network
scope = "network"
with tf.variable_scope(scope):
    # Estimation for unnormalized log probability
    Y = network(S, game.nx, game.ny) 
    # Estimation for probability
    P = tf.nn.softmax(Y, name = "softmax")
    # Target in integer
    W = tf.placeholder(tf.int32, shape = [None], name = "W")
    # Target in one-hot vector
    Y_= tf.one_hot(W, 3, name = "Y_")
    # Define loss and optimizer for value network
    loss, optimizer = network_optimizer(Y, Y_, alpha, scope)

### SAVER ###
saver = tf.train.Saver(max_to_keep = 0)

#####################################################################
"""                 TRAINING AND TESTING NETWORK                  """
#####################################################################

with tf.Session() as sess:
    ### DEFAULT SESSION ###
    sess.as_default()

    win1 = []; lose1 = []; tie1 = [];
    win2 = []; lose2 = []; tie2 = [];
 
    ### VARIABLE INITIALIZATION ###
    sess.run(tf.global_variables_initializer())
    
    # Load session
    saver.restore(sess, "./project3_task1.ckpt")

    ## Define parameters
    generation = 0
    n_test = 1

    print("Evaluating neural network against itself")
    r1 = np.zeros((n_test)) # randomness for player 1
    r2 = np.zeros((n_test))  # randomness for player 2
    s = game.play_games(P, r1, P, r2, n_test, nargout = 1)
    win1.append(s[0][0]); lose1.append(s[0][1]); tie1.append(s[0][2]);
    print(" net plays black: win=%6.4f, loss=%6.4f, tie=%6.4f" %\
        (win1[generation], lose1[generation], tie1[generation]))

    r1 = np.zeros((n_test))  # randomness for player 1
    r2 = np.zeros((n_test)) # randomness for player 2
    s = game.play_games(P, r1, P, r2, n_test, nargout = 1)
    win2.append(s[0][1]); lose2.append(s[0][0]); tie2.append(s[0][2]);
    print(" net plays white: win=%6.4f, loss=%6.4f, tie=%6.4f" %\
        (win2[generation], lose2[generation], tie2[generation]))

    ## Define parameters
    generation = 0
    n_test = 100000

    print("Evaluating neural network against random policy")
    generation = 1
    n_test = n_test_list[generation]
    r1 = np.zeros((n_test)) # randomness for player 1
    r2 = np.ones((n_test))  # randomness for player 2
    s = game.play_games(P, r1, [], r2, n_test, nargout = 1)
    win1.append(s[0][0]); lose1.append(s[0][1]); tie1.append(s[0][2]);
    print(" net plays black: win=%6.4f, loss=%6.4f, tie=%6.4f" %\
        (win1[generation], lose1[generation], tie1[generation]))

    r1 = np.ones((n_test))  # randomness for player 1
    r2 = np.zeros((n_test)) # randomness for player 2
    s = game.play_games([], r1, P, r2, n_test, nargout = 1)
    win2.append(s[0][1]); lose2.append(s[0][0]); tie2.append(s[0][2]);
    print(" net plays white: win=%6.4f, loss=%6.4f, tie=%6.4f" %\
        (win2[generation], lose2[generation], tie2[generation]))