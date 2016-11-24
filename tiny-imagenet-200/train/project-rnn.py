
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#Above code taken from the internet to download and export the standard MNIST database.

'''
At every time-step, we are handling 28 pixels. Since the MNIST database images are 28*28 pixels, each image is done in 28 timesteps.
'''

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 50
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_imagesteps = 28 # timesteps for a single image.
n_hidden1 = 128 # Number of features in first hidden layer
n_hidden2 = 32 # Number of features in second hidden layer
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_imagesteps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'h1': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
    'out' : tf.Variable(tf.random_normal([n_hidden2, n_classes]))

}
biases = {
    'h1' : tf.Variable(tf.random_normal([n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Change the data shape to fit the dimentions RNN works best with. Current shape= (batch_size. n_imageiterations, n_input) but we need to change it to (batch_size, n_input)
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_imagesteps, x)

    # Definition of a tensorflow LSTM cell.
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden1, forget_bias=1.0)

    # Output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    h2 = tf.matmul(outputs[-1], weights['h1']) + biases['h1']
    return tf.matmul(h2, weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Add reguralizer.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    iteration = 1
    # Keep training until reach max iterations
    while iteration * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_imagesteps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if iteration % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            acc = 100*acc
            print("Iteration " + str(iteration*batch_size) + ", Batch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy " + \
                  "{:.2f}".format(acc)) + "%"
        iteration += 1
    print("Training Finished!")

    # Calculate accuracy for the 1000 test images.
    test_len = 1000
    test_data = mnist.test.images[:test_len].reshape((-1, n_imagesteps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
