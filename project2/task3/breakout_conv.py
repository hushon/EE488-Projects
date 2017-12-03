import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from breakout_env import *
from breakout_ani import *
from wait import *
from time import sleep

## define Q network ##
lr = 0.5 
nh = 100 # adjust to avoid overfitting
ni = env.ny*env.nx*env.nf # size of input vector
no = env.na # size of output vector
depth = 16 # number of convolution filter
# parameters
n_episodes = 100
max_steps = 100
epsilon = 0.3 #epsilon-greedy factor
batch_size = 32 #size of a minibatch
gamma = 0.99 #discount factor
C = 30 # target network update frequency

class replayMemory:
    def __init__(self):
        self.current_size = 0
        self.arr_s = []
        self.arr_a = []
        self.arr_r = []
        self.arr_sn = []
        self.arr_term = []

    def store(self, s, a, r, sn, terminal):
        self.arr_s.append(s)
        self.arr_a.append(a)
        self.arr_r.append(r)
        self.arr_sn.append(sn)
        self.arr_term.append(terminal)
        self.current_size+=1

    def next_batch(self, batch_size):
        items = np.arange(0, self.current_size)
        if(self.current_size<=batch_size):
            index = items
            size = self.current_size
        else:
            index = np.random.choice(items, size=batch_size)
            size = batch_size
        batch_s = np.copy(np.array(self.arr_s)[index])
        batch_a = np.copy(np.array(self.arr_a)[index])
        batch_r = np.copy(np.array(self.arr_r)[index])
        batch_sn = np.copy(np.array(self.arr_sn)[index])
        batch_term = np.copy(np.array(self.arr_term)[index])
        return size, batch_s, batch_a, batch_r, batch_sn, batch_term

class minibatch: 
    pass


env = breakout_environment(5, 8, 3, 1, 2)

sess = tf.InteractiveSession()
### define Q network ##
#lr = 0.5 
#nh = 100 # adjust to avoid overfitting
#ni = env.ny*env.nx*env.nf # size of input vector
#no = env.na # size of output vector
#depth = 16 # number of convolution filter
x = tf.placeholder(tf.float32, shape=[None, ni])
y = tf.placeholder(tf.float32, shape=[None, no])
# Convolution layer
x_image = tf.reshape(x, [-1, env.ny, env.nx, env.nf])
W_conv = tf.Variable(name='W_conv', initial_value=tf.truncated_normal([2, 2, 2, depth], stddev=0.1))
b_conv = tf.Variable(name='b_conv', initial_value=tf.truncated_normal([depth], stddev=0.1))
h_conv = tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='VALID')
h_relu = tf.nn.relu(h_conv + b_conv)
h_relu_flat = tf.reshape(h_relu, [-1, 7*4*depth])
# Hidden layer
W_h = tf.Variable(name='W_h', initial_value=tf.truncated_normal(shape=[7*4*depth, nh], stddev=0.1))
b_h = tf.Variable(name='b_h', initial_value=tf.truncated_normal(shape=[nh], stddev=0.1))
# W_h = tf.Variable(tf.truncated_normal(shape=[ni, nh], stddev=0.1))
# b_h = tf.Variable(tf.truncated_normal(shape=[nh], stddev=0.1))
# W_h = tf.Variable(tf.random_uniform(shape=[ni, nh], minval=0.0, maxval=0.5))
# b_h = tf.Variable(tf.random_uniform(shape=[nh], minval=0.0, maxval=0.5))
h = tf.nn.relu(tf.matmul(h_relu_flat, W_h) + b_h)
# Output layer
W_o = tf.Variable(name='W_o', initial_value=tf.ones(shape=[nh, no]))
b_o = tf.Variable(name='b_o', initial_value=tf.ones(shape=[no]))
# W_o = tf.Variable(tf.random_uniform(shape=[nh, no], minval=0.0, maxval=0.5))
# b_o = tf.Variable(tf.random_uniform(shape=[no], minval=0.0, maxval=0.5))
Q=tf.matmul(h, W_o) + b_o
# cost function and optimizer
cost = tf.reduce_mean((y-Q)**2)
train_Q = tf.train.AdamOptimizer(lr).minimize(cost)
sess.run(tf.initialize_all_variables()) # initialize Q and Q_hat

## define Q_hat network ##
x_hat = tf.placeholder(tf.float32, shape=[None, ni])
x_image_hat = tf.reshape(x_hat, [-1,env.ny, env.nx, env.nf])
W_conv_hat = sess.run(W_conv)
b_conv_hat = sess.run(b_conv)
h_conv_hat = tf.nn.conv2d(x_image_hat, W_conv_hat, strides=[1, 1, 1, 1], padding='VALID')
h_relu_hat = tf.nn.relu(h_conv_hat + b_conv_hat)
h_relu_flat_hat = tf.reshape(h_relu_hat, [-1, 7*4*depth])
# W_h_hat = tf.constant(sess.run(W_h))
# b_h_hat = tf.constant(sess.run(b_h))
W_h_hat = sess.run(W_h)
b_h_hat = sess.run(b_h)
h_hat = tf.nn.relu(tf.matmul(h_relu_flat_hat, W_h_hat) + b_h_hat)
# W_o_hat = tf.constant(sess.run(W_o))
# b_o_hat = tf.constant(sess.run(b_o))
W_o_hat = sess.run(W_o)
b_o_hat = sess.run(b_o)
Q_hat=tf.matmul(h_hat, W_o_hat) + b_o_hat

## parameters
#n_episodes = 100
#max_steps = 100
#epsilon = 0.3 #epsilon-greedy factor
#batch_size = 32 #size of a minibatch
#gamma = 0.99 #discount factor
#C = 30 # target network update frequency

batch = minibatch()
memory = replayMemory() # initialize replay memory
step=0
# sess.run(tf.initialize_all_variables()) # initialize Q and Q_hat
for episode in range(n_episodes):
    s = env.reset() 
    episode_reward = 0
    step+=1
    print("===episode #", episode, "===")
    for t in range(max_steps): 
        if (np.random.rand() < epsilon): #with probability epsilon select a random action a_t
            a = np.random.randint(env.na)
        else: #otherwise select a_t = argmax(a, Q(phi(s_t), a; theta))
            q = Q.eval(feed_dict={x: np.reshape(s, [1, env.ny*env.nx*env.nf])})
#            print("action-value",q)
            a = np.argmax(q)
            

#        print("action",a)
        sn, r, terminal, _, _, _, _, _, _, _, _ = env.run(a - 1) # action a_t in emulator and observe reward r_t and image x_(t+1)
        memory.store(s, a, r, sn, terminal)
        
        batch.size, batch.s, batch.a, batch.r, batch.ns, batch.terminal = memory.next_batch(batch_size) ## debug this
        y_target = []
        for i in range(batch.size):
            current_Q = Q.eval(feed_dict={x: np.reshape(batch.s[i], [1, env.ny*env.nx*env.nf])})[0]
            if(batch.terminal[i]==1):
                a_i = batch.a[i]
                y_i = batch.r[i]
            else:
                q_i = Q_hat.eval(feed_dict={x_hat: np.reshape(batch.ns[i], [1, env.ny*env.nx*env.nf])})[0]
                a_i = np.argmax(q_i)
                y_i = batch.r[i] + gamma*np.max(q_i)
            current_Q[a_i] = y_i
            y_target.append(current_Q)
        train_Q.run(feed_dict={x: np.reshape(batch.s, [batch.size, env.ny*env.nx*env.nf]), y: np.array(y_target)})

        if(step%C==0): # every C steps set Q_hat to Q
            W_conv_hat = sess.run(W_conv)
            b_conv_hat = sess.run(b_conv)
            W_h_hat = sess.run(W_h)
            b_h_hat = sess.run(b_h)
            W_o_hat = sess.run(W_o)
            b_o_hat = sess.run(b_o)

        s=np.copy(sn)
        episode_reward+=r
        if(terminal==1): 
            print("steps:", t, "episode reward:", episode_reward, "terminal", terminal)
            break # if the episode reached terminal then jump to next episode


## Tensorflow Saver
saver = tf.train.Saver()
# saver.restore(sess, "./breakout.ckpt")
#save_path = saver.save(sess, "./breakout.ckpt")

# ani = breakout_animation(env, 20)
# ani.save('breakout.mp4', dpi=200)
# plt.show(block=False)
# wait('Press enter to quit')