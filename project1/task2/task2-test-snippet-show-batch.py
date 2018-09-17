# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 13:15:52 2017

@author: 
"""

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
batch = mnist.train.next_batch(1)

print("==========")
print("type of batch[0]: ", type(batch[0]))
print("shape of batch[0]: ", np.shape(batch[0]))
print("batch[0]: \n", batch[0])
print("==========")
print("type of batch[1]: ", type(batch[1]))
print("shape of batch[1]: ", np.shape(batch[1]))
print("batch[0]: \n", batch[1])

