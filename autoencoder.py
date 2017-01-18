from get_features import getLbp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

print 'Loading Data....'
x_train = np.load('Data/x_train.npy')
x_test = np.load('Data/x_test.npy')
x_train = np.squeeze(x_train)
# print len(x_train)
# sys.exit()


# Parameters
learning_rate = 0.01
training_epochs = 20


# Network Parameters
n_hidden_1 = 128
n_hidden_2 = 50
n_input = 256

# tf Graph input
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

# Building the encoder
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
y_pred = decoder_op
y_true = X

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()



Bsize = 256
Tsize = len(x_train)
Nbatch = int(Tsize/Bsize)
if Tsize/(Bsize*1.0) != Nbatch:
    Nbatch = Nbatch + 1
print 'Total Number of Batches:',Nbatch
nEpochs = 30

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(nEpochs):
        print "Epoch:",epoch+1,
        for i in range(Nbatch):
            if(i+1 == Nbatch):
                a = (i)*Bsize+1
                b = Tsize
            else:
                a = (i)*Bsize+1
                b = (i+1)*Bsize

            # print a,b
            batch_x = x_train[a:b]
            # print np.shape(x_train),np.shape(batch_x)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x})
        print "cost:",c
    print("Optimization Finished!")
    w,b = sess.run([weights, biases])
np.save('Data/ae-weights.npy',w)
np.save('Data/ae-biases.npy',b)
