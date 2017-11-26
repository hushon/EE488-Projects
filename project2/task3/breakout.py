import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from breakout_env import *
from breakout_ani import *
from wait import *

# define parameters
nx = 5   # number of pixels of screen = ny * nx
ny = 8
nb = 3   # number of rows of bricks
nt = 1   # gap at the top
nf = 2   # number of most recent frames at the input layer 

env = breakout_environment(nx, ny, nb, nt, nf)


### Define neural network model here ###
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
# saver.restore(sess, "./breakout.ckpt")
### training goes here ###
save_path = saver.save(sess, "./breakout.ckpt")


ani = breakout_animation(env, 20)
#ani.save('breakout.mp4', dpi=200)
plt.show(block=False)
wait('Press enter to quit')
 
