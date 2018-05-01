# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 18:58:58 2018

@author: Vishrut Sharma
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE_PX = 50
SLICE_COUNT = 20

n_classes = 2

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8

def conv3d(x,W):
    return tf.nn.conv3d(x,W, strides = [1,1,1,1,1], padding = 'SAME')

def maxpool3d(x):
    return tf.nn.max_pool3d(x, ksize = [1,2,2,2,1], strides = [1,2,2,2,1],
                            padding = 'SAME')

def convolutional_neural_network(x):
    # 5* 5 * 5 patches, 1 channel, 32 features to compute
    with tf.device('/cpu:0'):
        weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
                   #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
                   'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
                   #                                  64 features
                   'W_fc':tf.Variable(tf.random_normal([54080,1024])),
                   'out':tf.Variable(tf.random_normal([1024, n_classes]))}
    
        biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
                   'b_conv2':tf.Variable(tf.random_normal([64])),
                   'b_fc':tf.Variable(tf.random_normal([1024])),
                   'out':tf.Variable(tf.random_normal([n_classes]))}
    
        #                            image X      image Y        image Z
        x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])
    
        conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
        conv1 = maxpool3d(conv1)
    
    
        conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = maxpool3d(conv2)
    
        fc = tf.reshape(conv2,[-1, 54080])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
        fc = tf.nn.dropout(fc, keep_rate)
    
        output = tf.matmul(fc, weights['out'])+biases['out']

    return output

much_data = np.load('lungdata-50-50-20.npy')
# If you are working with the basic sample data, use maybe 2 instead of 100 here... you don't have enough data to really do this
train_data = much_data[:-100]
validation_data = much_data[-100:]

ep = []
epl = []
va = []

#tensorboard for graohs for accuracy 
def train_neural_network(x):
    
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction,labels = y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    hm_epochs = 10   
    with tf.Session() as sess:
        sess.run(init_op)
        successful_runs = 0
        total_runs = 0
        for epoch in range(hm_epochs):
            epoch_loss = 0
            ep.append(epoch)
        
            for data in train_data:
                total_runs += 1
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one 
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
                    pass
                    #print(str(e))
                        
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
            epl.append(epoch_loss/1000000000)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            accuracy_value = accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]})
            print('Accuracy:',accuracy_value)
            va.append(accuracy_value)
            
            
        print('Done. Finishing accuracy:')
        print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
        
        print('fitment percent:',successful_runs/total_runs)
        saver.save(sess, './model.ckpt')

train_neural_network(x)
def loss_graph():
    plt.plot(ep,epl)
    plt.axis([0,10,0,100])
    plt.show()
    
def accuracy_graph():
    plt.plot(ep,va) 
    plt.axis([0,10,0,1])
    plt.show()
    
    
"""
To import a pretrained model
    
    """
loss_graph()
accuracy_graph()    