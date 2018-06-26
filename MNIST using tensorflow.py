# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 17:06:25 2018

@author: dhruv_dzb8kxe
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
mnist

mnist.train.num_examples

mnist.test.num_examples

mnist.test.labels.shape
import matplotlib.pyplot as plt

single_image = mnist.train.images[2].reshape(28,28)
plt.imshow(single_image,cmap='gist_gray')


single_image.min()
single_image.max()



#placeholders
x = tf.placeholder(tf.float32,shape=[None,784])

#variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#create graph operations
y = tf.matmul(x,W) + b
#loss functions
y_true = tf.placeholder(tf.float32,shape=[None,10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))
#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)
#creaet session

init = tf.global_variables_initializer()


with tf.Session()  as sess:
    sess.run(init)
    
    for step in range(1000):
        batch_x,batch_y = mnist.train.next_batch(100)
        
        sess.run(train,feed_dict={x:batch_x,y:batch_y})

#EVALUATE THE MODEL
correct_prediction = tf.equal( tf.argmax(y,1), tf.argmax(y_true,1))
#TRUE FALSE --->>>[1,0,1,0,0]

acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# PREDICTED DATA [3,4] TRUE [3,9] FALSE

#[TRUE ,FALSE]

#[1.0,0.0]
print(mnist.train.images.shape, mnist.test.labels.shape)
#0.5   <- average of 1.0 + 0.0   / 2  = 0.5
feed_dict={x:mnist.test.images,y_true:mnist.test.labels}
with tf.Session()  as sess:
    sess.run(init)
    
    for step in range(1000):
        batch_x,batch_y = mnist.train.next_batch(100)
        sess.run(acc,feed_dict)

    print(sess.run(acc,feed_dict))










