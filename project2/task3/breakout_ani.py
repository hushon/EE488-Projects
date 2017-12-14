# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/19/2017

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import tensorflow as tf




class breakout_animation(animation.TimedAnimation):
    def __init__(self, env, max_steps, nh, depth, frames_per_step = 5):
        self.nh = nh
        self.depth = depth

        self.env = env
        self.max_steps = max_steps

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        self.objs = []

        # boundary
        w = 0.1
        ax.plot([-w,-w,env.nx+w,env.nx+w],[0,env.ny+w,env.ny+w,0],'k-',linewidth=5)

        # bricks
        wb = 0.05
        self.bricks = []
        self.brick_colors = [['red'], ['blue','red'], ['blue','green','red'], ['blue','green','yellow','red'], ['blue','green','yellow','orange','red'], \
            ['purple','blue','green','yellow','brown','orange','red'], ['purple','blue','green','yellow','brown','orange','red']]    # add more colors if needed
        for y in range(self.env.nb):
            b = []
            yp = y + (self.env.ny - self.env.nt - self.env.nb)
            for x in range(self.env.nx):
                b.append(patches.Rectangle((x + wb, yp + wb), 1-2*wb, 1-2*wb, edgecolor='none', facecolor=self.brick_colors[self.env.nb-1][y]))
                ax.add_patch(b[x])
                self.objs.append(b[x])
            self.bricks.append(b)
 
        # ball
        self.ball = patches.Circle(env.get_ball_pos(0.), radius = 0.15, color = 'red')
        ax.add_patch(self.ball)
        self.objs.append(self.ball)

        # score text
        self.text = ax.text(0.5 * env.nx, 0, '', ha='center')
        self.objs.append(self.text)

        # game over text
        self.gameover_text = ax.text(0.5 * env.nx, 0.5 * env.ny, '', ha='center')
        self.objs.append(self.gameover_text)

        self.frames_per_step = frames_per_step
        self.total_frames = self.frames_per_step * self.max_steps

        # paddle
        self.paddle = patches.Rectangle((env.p, 0.5), 1, 0.5, edgecolor='none', facecolor='red')
        ax.add_patch(self.paddle)

        # for early termination of animation
        self.iter_objs = []
        self.iter_obj_cnt = 0

        # interval = 50msec
        animation.TimedAnimation.__init__(self, fig, interval=50, repeat=False, blit=False)


    def get_Q(self, env):
        ##===== for Q network ==========
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        nh = self.nh # adjust to avoid overfitting
        ni = env.ny*env.nx*env.nf # size of input vector
        no = env.na # size of output vector
        depth = self.depth
        x = tf.placeholder(tf.float32, shape=[None, ni])
        x_image = tf.reshape(x, [-1, env.ny, env.nx, env.nf])
        W_conv = tf.get_variable(name='W_conv', shape=[2, 2, 2, depth])
        b_conv = tf.get_variable(name='b_conv', shape=[depth])
        h_conv = tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='VALID')
        h_relu = tf.nn.relu(h_conv + b_conv)
        h_relu_flat = tf.reshape(h_relu, [-1, 7*4*depth])
        # Hidden layer
        W_h = tf.get_variable(name='W_h', shape=[7*4*depth, nh])
        b_h = tf.get_variable(name='b_h', shape=[nh])
        h = tf.nn.relu(tf.matmul(h_relu_flat, W_h) + b_h)
        # Output layer
        W_o = tf.get_variable(name='W_o', shape=[nh, no])
        b_o = tf.get_variable(name='b_o', shape=[no])
        Q=tf.matmul(h, W_o) + b_o
        saver = tf.train.Saver()
        saver.restore(sess, "./breakout.ckpt")

        return np.argmax(Q.eval(feed_dict={x: np.reshape(env.s, [1, env.ny*env.nx*env.nf])})[0]) - 1 ## testing..
        ##==============================



    def _draw_frame(self, k):

        
        if self.terminal:
            return
        if k == 0:
            self.iter_obj_cnt -= 1
        if k % self.frames_per_step == 0:
            # self.a = np.random.randint(3) - 1
            self.a = self.get_Q(self.env)
            self.p = self.env.p
            self.pn = min(max(self.p + self.a, 0), self.env.nx - 1)

        t = (k % self.frames_per_step) * 1. / self.frames_per_step
        self.ball.center = self.env.get_ball_pos(t)
        self.paddle.set_x(t * self.pn + (1-t) * self.p)

        if k % self.frames_per_step == self.frames_per_step - 1:
            sn, reward, terminal, p0, p, bx0, by0, vx0, vy0, rx, ry = self.env.run(self.a)
            self.sum_reward += reward
            if reward > 0.:
                self.bricks[ry][rx].set_facecolor('none')
                self.text.set_text('Score: %d' % self.sum_reward)
            if terminal:
                self.terminal = terminal
                self.gameover_text.set_text('Game Over')
                for _ in range(self.total_frames - k - 1):
                    self.iter_objs[self.iter_obj_cnt].next()     # for early termination of animation (latest iterator is used first)

        self._drawn_artists = self.objs

    def new_frame_seq(self):
        iter_obj = iter(range(self.total_frames))
        self.iter_objs.append(iter_obj)
        self.iter_obj_cnt += 1
        return iter_obj

    def _init_draw(self):
        # _ = self.env.reset()
        self.s = self.env.reset()
        self.sum_reward = 0.
        self.p = self.env.p    # current paddle position
        self.pn = self.p       # next paddle position
        self.a = 0             # action
        self.terminal = 0

        for y in range(self.env.nb):
            for x in range(self.env.nx):
                self.bricks[y][x].set_facecolor(self.brick_colors[self.env.nb-1][y])

        self.ball.center = self.env.get_ball_pos(0.)
        self.paddle.set_x(self.p)

        self.text.set_text('Score: 0')
        self.gameover_text.set_text('')

