# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017

import numpy as np
import matplotlib.pyplot as plt
from qlearning import *

class linear_environment:
    def __init__(self):
        self.n_states = 21       # number of states
        self.n_actions = 2      # number of actions
        self.next_state = np.array([[0,0],[0,2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,8],[7,9],[8,10],[9,11],[10,12],[11,13],[12,14],[13,15],[14,16],[15,17],[16,18],[17,19],[18,20],[20,20]], dtype=np.int)    # next_state
        self.reward = np.array([[0.,0.],[1.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,1.],[0.,0.]])            # reward for each (state,action)
        self.terminal = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], dtype=np.int)                          # 1 if terminal state, 0 otherwise
        self.init_state = 10     # initial state


# an instance of the environment
env = linear_environment()

#n_episodes = 1      # number of episodes to run
# n_episodes = 5      # number of episodes to run
#n_episodes = 100      # number of episodes to run
n_episodes = 1000      # number of episodes to run
max_steps = 1000        # max. # of steps to run in each episode
alpha = 0.2          # learning rate
gamma = 0.9          # discount factor

class epsilon_profile: pass
epsilon = epsilon_profile()
epsilon.init = 1.    # initial epsilon in e-greedy
epsilon.final = 0.   # final epsilon in e-greedy
#epsilon.final = 1.   # final epsilon in e-greedy
epsilon.dec_episode = 1. / n_episodes  # amount of decrement in each episode
epsilon.dec_step = 0.                  # amount of decrement in each step

Q, n_steps, sum_rewards = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon)
print('Q(s,a): ')
print(Q)
# for k in range(n_episodes):
#     print('Reward of episode #%2d: %.2f' % (k, sum_rewards[k]))
print("Number of training episodes:", n_episodes)
print("Avg training steps:", np.mean(n_steps))

test_n_episodes = 1  # number of episodes to run
test_max_steps = 1000   # max. # of steps to run in each episode
test_epsilon = 0.    # test epsilon
test_n_steps, test_sum_rewards, s, a, sn, r = Q_test(Q, env, test_n_episodes, test_max_steps, test_epsilon)
print("test_sum_rewards[0]:", test_sum_rewards[0])
print("test_n_steps[0]:", test_n_steps[0])

plt.figure(1)
plt.plot(np.arange(0, np.size(n_steps)), n_steps)
plt.show()