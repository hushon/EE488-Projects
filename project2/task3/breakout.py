import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from breakout_env import *
from breakout_ani import *
from wait import *

class replayMemory:
	def __init__(self, N):
		self.current_size = 0
		self.arr_s = []
		self.arr_a = []
		self.arr_r = []
		self.arr_sn = []
		self.arr_term = []
		self.N = N # this does not bound the size of replayMemory

	def store(self, s, a, r, sn, terminal):
		self.arr_s.append(s)
		self.arr_a.append(a)
		self.arr_r.append(r)
		self.arr_sn.append(sn)
		self.arr_term.append(terminal)
		self.current_size+=1
		if(self.current_size>=self.N): print("replay memory current_size:", current_size)

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
sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# saver.restore(sess, "./breakout.ckpt")

## define Q network ##
lr = 0.001 
nh = 512 # adjust to avoid overfitting
ni = env.ny*env.nx*env.nf # size of input vector
no = env.na # size of output vector
x = tf.placeholder(tf.float32, shape=[None, ni])
y = tf.placeholder(tf.float32, shape=[None, no])
# Hidden layer
W_h = tf.Variable(tf.truncated_normal(shape=[ni, nh], stddev=0.1))
b_h = tf.Variable(tf.truncated_normal(shape=[nh], stddev=0.1))
h = tf.nn.relu(tf.matmul(x, W_h) + b_h)
# Output layer
W_o = tf.Variable(tf.truncated_normal(shape=[nh, no], stddev=0.1))
b_o = tf.Variable(tf.truncated_normal(shape=[no], stddev=0.1))
Q=tf.matmul(h, W_o) + b_o
# cost function and optimizer
cost = tf.reduce_mean((y-Q)**2)
train_Q = tf.train.AdamOptimizer(lr).minimize(cost)
sess.run(tf.initialize_all_variables()) # initialize Q and Q_hat

## define Q_hat network ##
x_hat = tf.placeholder(tf.float32, shape=[None, ni])
W_h_hat = tf.constant(sess.run(W_h))
b_h_hat = tf.constant(sess.run(b_h))
h_hat = tf.nn.relu(tf.matmul(x_hat, W_h_hat) + b_h_hat)
W_o_hat = tf.constant(sess.run(W_o))
b_o_hat = tf.constant(sess.run(b_o))
Q_hat=tf.matmul(h_hat, W_o_hat) + b_o_hat


# parameters
n_episodes = 1000
max_steps = 1000
epsilon = 0.2 #epsilon-greedy factor
batch_size = 32 #size of a minibatch
gamma = 0.99 #discount factor
C = 100 # target network update frequency

batch = minibatch()

memory = replayMemory(N=10000) # initialize replay memory
sess.run(tf.initialize_all_variables()) # initialize Q and Q_hat
for episode in range(n_episodes):
	s = env.reset() 
	for t in range(max_steps): 
		if (np.random.rand() < epsilon): #with probability epsilon select a random action a_t
			a = np.random.randint(env.na)-1
		else: #otherwise select a_t = argmax(a, Q(phi(s_t), a; theta))
			q = Q.eval(feed_dict={x: np.reshape(s, [1, env.ny*env.nx*env.nf])})
			a = np.argmax(q)

		sn, r, terminal, _, _, _, _, _, _, _, _ = env.run(a - 1) # action a_t in emulator and observe reward r_t and image x_(t+1)
		memory.store(s, a, r, sn, terminal)
		
		batch.size, batch.s, batch.a, batch.r, batch.ns, batch.terminal = memory.next_batch(batch_size) ## debug this
		y_target = []
		for i in range(batch.size):
			current_Q = Q.eval(feed_dict={x: np.reshape(batch.s[i], [1, env.ny*env.nx*env.nf])})[0]
			if(batch.terminal[i]==1):
				a_i = batch.a[i]
				y_i = batch.r[i]
				current_Q[a_i] = y_i
			else:
				q_i = Q_hat.eval(feed_dict={x_hat: np.reshape(batch.ns[i], [1, env.ny*env.nx*env.nf])})[0]
				a_i = np.argmax(q_i)
				y_i = batch.r[i] + gamma*np.max(q_i)
				current_Q[a_i] = y_i
			y_target.append(current_Q)
		train_Q.run(feed_dict={x: np.reshape(batch.s, [batch.size, env.ny*env.nx*env.nf]), y: np.array(y_target)}) # 이부분 y가 잘못되었는데..

		if(t%C==0): # every C steps set Q_hat to Q
			W_h_hat = tf.constant(sess.run(W_h))
			b_h_hat = tf.constant(sess.run(b_h))
			W_o_hat = tf.constant(sess.run(W_o))
			b_o_hat = tf.constant(sess.run(b_o))
			print("step:", t, "reward:", r)

		s=np.copy(sn)
		if(terminal==1): break # if the episode reached terminal then jump to next episode


# save_path = saver.save(sess, "./breakout.ckpt")

ani = breakout_animation(env, 20)
#ani.save('breakout.mp4', dpi=200)
plt.show(block=False)
wait('Press enter to quit')