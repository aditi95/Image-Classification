import os, sys
from PIL import Image
import numpy as np
from sklearn import preprocessing
import tensorflow as tf 

enc = preprocessing.OneHotEncoder()
y = []

# im = Image.open("fish.JPEG")
# x = 3
# y = 4

# im.load()
# data = np.asarray( im, dtype="int32" )
# print data
x = np.empty((0, 64,64,3), dtype="int32")
#y = np.empty((0, num_labels), float)
num_labels = 2
count = 0
mapping = {}
for i in os.listdir('.'):
	mapping[count] = i
	if i == 'project.py':
		continue 
	for j in os.listdir(i+'/images'):
		im = Image.open(i+'/images/'+j)
		im.load()
		data = np.asarray( im, dtype="int32" )
		#print np.shape(np.shape(data))[0]
		if np.shape(np.shape(data))[0] == 2:
			d = np.empty((64, 64, 3), dtype="int32")
			d[:,:,0] = data
			d[:,:,1] = data
			d[:,:,2] = data
			data = d
		x = np.append(x,[data],axis = 0)
		temp = []
		temp.append(count)
		y.append(temp)
		print np.shape(y)
	count = count + 1
	if count >= num_labels:
		break

enc.fit(y)
y_train = enc.transform(y).toarray()
num = np.shape(y_train)[1]
print np.shape(y_train)

sess = tf.InteractiveSession()

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#print(mnist.train.labels)
#print(y_train)

num_inputs = 64*64*3
hidden_layer_1 = 1024
hidden_layer_2 = 128
#getting images

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')





X = tf.placeholder("float", shape=[None,64,64,3])
Y = tf.placeholder("float", shape=[None,num_labels])


W1 = weight_variable([7,7,3,50])
b1 = bias_variable([50])
#x_image = tf.reshape(X, [-1,64,64,3])
h_conv1 = tf.nn.relu(conv2d(X, W1) + b1)
h_pool1 = max_pool_2x2(h_conv1)


W2 = weight_variable([5,5,50,50])
b2 = bias_variable([50])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b2)
h_pool2 = max_pool_2x2(h_conv2)


W3 = weight_variable([5,5,50,50])
b3 = bias_variable([50])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W3) + b3)
h_pool3 = max_pool_2x2(h_conv3)


W4 = weight_variable([5,5,50,128])
b4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W4) + b4)
h_pool4 = max_pool_2x2(h_conv4)

X1 = tf.reshape(h_pool4, [-1,8*8*128])
Wfc1 = weight_variable([8*8*128, 256])
bfc1 = bias_variable([256])
hfc1 = tf.nn.relu(tf.matmul(X1, Wfc1) + bfc1)
keep_prob1 = tf.placeholder(tf.float32)
hfc1_drop = tf.nn.dropout(hfc1, keep_prob1)

Wfc2 = weight_variable([256, 64])
bfc2 = bias_variable([64])
hfc2 = tf.nn.relu(tf.matmul(hfc1_drop, Wfc2) + bfc2)
keep_prob2 = tf.placeholder(tf.float32)
hfc2_drop = tf.nn.dropout(hfc2, keep_prob2)

Wfc3 = weight_variable([64, num_labels])
bfc3 = bias_variable([num_labels])
y_conv = tf.matmul(hfc2_drop, Wfc3) + bfc3


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, Y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())


  
for epoch in range(100):
  #batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={X: x, Y: y_train, keep_prob1: 0.5, keep_prob2: 0.5})  
    train_accuracy = accuracy.eval(feed_dict={X:x, Y: y_train, keep_prob1:1.0, keep_prob2: 1.0})
    print("step %d, training accuracy %g"%(epoch, train_accuracy))
      #prediction = correct_prediction.eval(feed_dict={X:x_test, Y: y_test})
      #print(prediction)

saver = tf.train.Saver([W1, b1, W2, b2, W3, b3, Wfc1, bfc1, Wfc2, bfc2, Wfc3, bfc3])
saver.save(session, "hello.chk")