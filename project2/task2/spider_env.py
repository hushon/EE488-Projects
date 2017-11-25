# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/12/2017

import numpy as np

# spider environment
class spider_environment:
    def __init__(self):
        leg_rb = baby_spider_environment()
        leg_lb = baby_spider_environment()
        leg_rf = baby_spider_environment()
        leg_lf = baby_spider_environment()
        bitmask_rb = 0b11000000
        bitmask_lb = 0b00110000
        bitmask_rf = 0b00001100
        bitmask_lf = 0b00000011
        self.n_states = 256
        self.n_actions = 256
        self.reward = np.zeros([self.n_states, self.n_actions])
        self.terminal = np.zeros(self.n_states, dtype=np.int)
        self.next_state = np.zeros([self.n_states, self.n_actions], dtype=np.int)
        self.init_state = 0b00001010
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.next_state[s,a] = (leg_rb.next_state[(s&bitmask_rb)>>6,(a&bitmask_rb)>>6]<<6)+(leg_lb.next_state[(s&bitmask_lb)>>4,(a&bitmask_lb)>>4]<<4)+(leg_rf.next_state[(s&bitmask_rf)>>2,(a&bitmask_rf)>>2]<<2)+(leg_lf.next_state[(s&bitmask_lf),(a&bitmask_lf)])
                
                LTACDASDAAAIT = 0b11111111 - (s|self.next_state[s,a]) # bitwise NOR of s and self.next_state[s,a]; indicates legs that are currently down and still down after an action is taken.
                total_down = (LTACDASDAAAIT>>6)%2 + (LTACDASDAAAIT>>4)%2 + (LTACDASDAAAIT>>2)%2 + (LTACDASDAAAIT)%2
                total_force = leg_rb.reward[(s&bitmask_rb)>>6,(a&bitmask_rb)>>6] + leg_lb.reward[(s&bitmask_lb)>>4,(a&bitmask_lb)>>4] + leg_rf.reward[(s&bitmask_rf)>>2,(a&bitmask_rf)>>2] + leg_lf.reward[(s&bitmask_lf),(a&bitmask_lf)]
                if(total_down==0):
                    self.reward[s,a] = 0
                elif(total_down>=3):
                    self.reward[s,a] = total_force/total_down
                elif(total_down==2):
                    if((((LTACDASDAAAIT>>4)%2) & ((LTACDASDAAAIT>>2)%2)==1) or (((LTACDASDAAAIT>>6)%2) & ((LTACDASDAAAIT)%2)==1) ):
                        self.reward[s,a] = total_force/total_down
                else:
                    self.reward[s,a] = 0.25*(total_force/total_down)
                 


class baby_spider_environment: 
    def __init__(self):
        self.n_states = 4         # number of states: leg up/down, forward/backward
        self.n_actions = 4        # number of actions 
        self.reward = np.zeros([self.n_states, self.n_actions])
        self.terminal = np.zeros(self.n_states, dtype=np.int)          # 1 if terminal state, 0 otherwise
        self.next_state = np.zeros([self.n_states, self.n_actions], dtype=np.int)        # next_state
        self.init_state = 0b00    # initial state
        transition = [[1,0,2,0],[1,0,3,1],[3,2,2,0],[3,2,3,1]]
        for s in range(4):
            up = s & 1
            fw = (s >> 1) & 1
            for a in range(4):
                action_up = (a & 3) == 0
                action_dn = (a & 3) == 1
                action_fw = (a & 3) == 2
                action_bw = (a & 3) == 3
                self.next_state[s,a] = transition[s][a]
                self.reward[s,a] = (up == 0 and fw == 1 and action_bw == 1) - (up == 0 and fw == 0 and action_fw == 1)

