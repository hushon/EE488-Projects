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
        self.n_states = 256
        self.n_actions = 256
        self.reward = np.zeros([self.n_states, self.n_actions])
        self.terminal = np.zeros(self.n_states, dtype=np.int)
        self.next_state = np.zeros([self.n_states, self.n_actions], dtype=np.int)
        self.init_state = 0b00001010
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # strip the state and action bitstrings into each legs
                s_rb = (s>>6)%4
                s_lb = (s>>4)%4
                s_rf = (s>>2)%4
                s_lf = (s>>0)%4
                a_rb = (a>>6)%4
                a_lb = (a>>4)%4
                a_rf = (a>>2)%4
                a_lf = (a>>0)%4

                # evaluate next_state
                self.next_state[s,a] = leg_rb.next_state[s_rb,a_rb]*64+leg_lb.next_state[s_lb,a_lb]*16+leg_rf.next_state[s_rf,a_rf]*4+leg_lf.next_state[s_lf,a_lf]
                
                # evaluate if a leg is currently down and still down after the action a
                # value is 1 if true
                states_NOR = 0b11111111 - (s|self.next_state[s,a]) # bitwise NOR of state and next_state.
                rb_down = (states_NOR>>6)%2
                lb_down = (states_NOR>>4)%2
                rf_down = (states_NOR>>2)%2
                lf_down = (states_NOR>>0)%2
                # evaluate total down
                total_down = rb_down+lb_down+rf_down+lf_down
                # evaluate total force
                total_force = leg_rb.reward[s_rb,a_rb] + leg_lb.reward[s_lb,a_lb] + leg_rf.reward[s_rf,a_rf] + leg_lf.reward[s_lf,a_lf]

                # evaluate reward
                if(total_down==0): 
                    self.reward[s,a] = 0
                elif(total_down>=3): 
                    self.reward[s,a] = total_force/total_down
                elif(total_down==2 and (rb_down*lf_down==1 or lb_down*rf_down==1)):
                    self.reward[s,a] = total_force/total_down
                    # if(s==0b01001010 and a==0b01011100): print(total_force, total_down, self.reward[s,a])
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

