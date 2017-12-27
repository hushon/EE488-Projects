# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)

# define activation function
def Phi(x): 
    t = np.copy(x)
    t[0] = np.tanh(x[0])
    t[1] = np.tanh(x[1]) # take tanh(x) to each entry
    return t

# define gradient of the activation function
def PhiPrime(x): 
    t = np.copy(x)
    t[0] = 1-np.square(np.tanh(x[0]))
    t[1] = 1-np.square(np.tanh(x[1]))
    return t

# Start of Algorithm
def TrainingAlgorithm(l, a, x, y, MaxIter): 
    # initialize weight array by w_init
    w_init = 5.
    w = np.full(l, w_init)
    
    # initialize l by 2 matrices u, h, g
    u = np.zeros((l, 2))
    h = np.zeros((l, 2))
    g = np.zeros((l, 2))
    m = 2 # number of training set
        
    # nvm, just for the cost function plot -1
    cost = np.zeros(MaxIter)
    
    for iter in range(0, MaxIter):
        
        ## for debugging, printing h and g after completing first iteration
        if iter==1:
            print("**Value of arrays after first iteration**")
            print("u=", u)
            print("w=", w)
            print("h=", h)
            print("g=", g)
            np.savetxt("ps2-1b-h.csv", h, delimiter=",")
            np.savetxt("ps2-1b-g.csv", g, delimiter=",")

        
        # beginning of forward propagation
        for k in range(0, l):
            if k==0:
                u[k] = np.multiply(x, w[k])
                h[k] = Phi(u[k])
                continue
            u[k] = np.multiply(h[k-1], w[k])
            h[k] = Phi(u[k])            
                
        # nvm, just for the cost function plot -2
        cost[iter] = np.square(np.linalg.norm(y-h[l-1]))/(2*m)
        
        # beginning of backpropagation
        g[l-1] = np.multiply(PhiPrime(u[l-1]), h[l-1]-y)/m # np.multiply() does element-wise multiplication
        for k in range(l-2, -1, -1):
            g[k] = np.multiply(PhiPrime(u[k]), g[k+1])*w[k+1]
            
        # beginning of gradient descent
        for k in range(0, l):
            w[k] = w[k]-a*np.dot(h[k-1], g[k])
    
    # nvm, just for the cost function plot-3
    plt.plot(np.arange(0, MaxIter), cost)
    plt.show()
    print("Final cost function: cost[MaxIter-1]=", cost[MaxIter-1]) # for printing final cost. this being zero means training is successful.
    print("Outcome of output layer after training: h[l-1]=", h[l-1]) ## debug: print final layer output##
    return w # return weight array

# define training algorithm parameters
l = 100 # number of layers
a = 0.1 # learning rate of gradient descent
MaxIter = 1000 # number of forward & backpropagation iterations

# training dataset x and y
x = np.array([-0.5, 0.5])
y = np.array([-0.5, 0.5])

# run algorithm
w = TrainingAlgorithm(l, a, x, y, MaxIter)
print("**Training Results**")
print("Weight of each Layer: w=", w)

####씨뽤됏다####