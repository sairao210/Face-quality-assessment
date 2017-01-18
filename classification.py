import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

# Load the required input data
x_train = np.load('Data/x_train.npy')
x_test = np.load('Data/x_test.npy')
x_train = np.squeeze(x_train)
x_test = np.squeeze(x_test)

lab_train = np.load('Data/lab_train.npy')
lab_test = np.load('Data/lab_test.npy')
lab_train = np.squeeze(lab_train)
lab_test = np.squeeze(lab_test)

# Load weights and biases of the autoencoder trained previously
w = np.load('Data/ae-weights.npy')
b = np.load('Data/ae-biases.npy')

# Parameters
learning_rate = 0.001
training_epochs = 20

# Network Parameters
n_hidden_1 = 128
n_hidden_2 = 50
n_input = 256
n_classes = 3

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'h1': tf.Variable(w[()]['encoder_h1']),
    'h2': tf.Variable(w[()]['encoder_h2']),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(b[()]['encoder_b1']),
    'b2': tf.Variable(b[()]['encoder_b2']),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


pred = multilayer_perceptron(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

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

        avg_cost = 0.
        for i in range(Nbatch):
            if(i+1 == Nbatch):
                a = (i)*Bsize+1
                b = Tsize
            else:
                a = (i)*Bsize+1
                b = (i+1)*Bsize

            # print a,b
            batch_x = x_train[a-1:b]
            batch_y = lab_train[a-1:b]
            # print np.shape(x_train),np.shape(batch_x)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            avg_cost = avg_cost + c

        print "cost:",avg_cost/(Nbatch*1.0)
    print("Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", sess.run(accuracy, feed_dict = {x: x_test, y: lab_test}))