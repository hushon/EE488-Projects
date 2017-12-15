import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

def exclude_digit(batch, digit): # eliminates given digit from batch
    batch_size = batch[1].shape[0]
    temp_x = batch[0]
    temp_y = batch[1]
    for i in range(batch_size-1, -1, -1):
        if(temp_y[i][digit]==1):
            temp_x = np.delete(temp_x, obj=i, axis=0)
            temp_y = np.delete(temp_y, obj=i, axis=0)
    temp_y = np.delete(temp_y, obj=digit, axis=1)
    return (temp_x, temp_y)

##=====First CNN with output neurons 1~9======
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 9])

# Convolutional layer
x_image = tf.reshape(x, [-1,28,28,1])
W_conv = tf.Variable(tf.truncated_normal([5, 5, 1, 30], stddev=0.1))
b_conv = tf.Variable(tf.constant(0.1, shape=[30]))
h_conv = tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='VALID')
h_relu = tf.nn.relu(h_conv + b_conv)
h_pool = tf.nn.max_pool(h_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully-connected layer
W_fc1 = tf.Variable(tf.truncated_normal([12 * 12 * 30, 500], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[500]))
h_pool_flat = tf.reshape(h_pool, [-1, 12*12*30])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

# Output layer
W_fc2 = tf.Variable(tf.truncated_normal([500, 9], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[9]))
y_hat=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# Train and Evaluate the Model
cross_entropy = - tf.reduce_sum(y_*tf.log(y_hat))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# prepare validation and test set
val_set = exclude_digit( (mnist.validation.images, mnist.validation.labels), digit=0 )
test_set = exclude_digit( (mnist.test.images, mnist.test.labels), digit=0 )

sess.run(tf.initialize_all_variables())
print("=================================")
print("CNN with modified MNIST set classifications 1~9")
print("|Epoch\tBatch\t|Train\t|Val\t|")
print("|===============================|")
for j in range(5):
    for i in range(550):
        batch = exclude_digit( mnist.train.next_batch(100), digit=0 )
        ##========데이터 변조 코드==========
#        digit = 0
#        temp_x = batch[0]
#        temp_y = batch[1]
#        size = 100
#        # eliminate data that corresponds to digit 0 from the batch
#        for k in range(size-1, -1, -1):   
#            if(temp_y[k][digit]==1): # scan through batch[1] and if 0 label is activated, delete corresponding row from batch[0]
#                temp_x = np.delete(temp_x, obj=k, axis=0)
#                temp_y = np.delete(temp_y, obj=k, axis=0)
#        temp_y = np.delete(temp_y, obj=digit, axis=1) # label 벡터의 길이가 10에서 9로 바뀜
#         그리고 feed_dict={x: temp_x, y:temp_y}
        ##=================================   
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        if i%50 == 49:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
            val_accuracy = accuracy.eval(feed_dict=\
                {x: val_set[0], y_:val_set[1]})
            print("|%d\t|%d\t|%.4f\t|%.4f\t|"%(j+1, i+1, train_accuracy, val_accuracy))
print("|===============================|")
test_accuracy = accuracy.eval(feed_dict=\
    {x: test_set[0], y_:test_set[1]})
print("test accuracy=%.4f"%(test_accuracy))



'''
##=====second CNN with output neurons 0~9======
### 데이터는 변조하지 않은 minst 를 사용한다.
###  (완료)그다음으로 같은 뉴럴네트워크를 가져와 output neuron을 하나 더 추가한다.
###  (완료)히든레이어의 파라미터값은 위의 레이어에서 그대로 가져올것!
### 아웃풋 레이어만 새로 만들어 학습시킨다.
### 이 네트워크의 accuracy는?
##  (완료)해야할거: 어떻게 아웃풋 레이어만 학습하도록 설정하나?

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Convolutional layer
## 학습한 파라미터를 그대로 가져오기
x_image = tf.reshape(x, [-1,28,28,1])
#W_conv = tf.Variable(tf.truncated_normal([5, 5, 1, 30], stddev=0.1))
#b_conv = tf.Variable(tf.constant(0.1, shape=[30]))
W_conv = sess.run(W_conv)
b_conv = sess.run(b_conv)
h_conv = tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='VALID')
h_relu = tf.nn.relu(h_conv + b_conv)
h_pool = tf.nn.max_pool(h_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Fully-connected layer
## 학습한 파라미터를 그대로 가져오기
#W_fc1 = tf.Variable(tf.truncated_normal([12 * 12 * 30, 500], stddev=0.1))
#b_fc1 = tf.Variable(tf.constant(0.1, shape=[500]))
W_fc1 = sess.run(W_fc1)
b_fc1 = sess.run(b_fc1)
h_pool_flat = tf.reshape(h_pool, [-1, 12*12*30])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

# Output layer
## 0을 판별하는 아웃풋 뉴런을 추가
W_fc2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_hat=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# Train and Evaluate the Model
cross_entropy = - tf.reduce_sum(y_*tf.log(y_hat))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())
print("=================================")
print("CNN with MNIST classifications 1~9")
print("|Epoch\tBatch\t|Train\t|Val\t|")
print("|===============================|")
for j in range(5):
    for i in range(550):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        if i%50 == 49:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
            val_accuracy = accuracy.eval(feed_dict=\
                {x: mnist.validation.images, y_:mnist.validation.labels})
            print("|%d\t|%d\t|%.4f\t|%.4f\t|"%(j+1, i+1, train_accuracy, val_accuracy))
print("|===============================|")
test_accuracy = accuracy.eval(feed_dict=\
    {x: mnist.test.images, y_:mnist.test.labels})
print("test accuracy=%.4f"%(test_accuracy))
'''