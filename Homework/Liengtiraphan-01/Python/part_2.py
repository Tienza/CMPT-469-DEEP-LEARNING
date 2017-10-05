import tensorflow as tf
import time

start_time = time.time()
#Start interactive session
sess = tf.InteractiveSession()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

width = 28 #Width of the image in pixels
height = 28 #Height of the image in pixels
flat = width * height #Number of pixels in one image
class_output = 10 #Numper of possible classifications for the problem

x = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

#Converting images of the data set to tensors
#The input image is 28x28 pixels, 1 channel(grayscale), first parameter is the batch number
#It can be of any size, so we set it to -1
x_image = tf.reshape(x, [-1,28,28,1])
x_image

#Convolutional Layer 1
#The size of the filter/kernel is 5x5; Input channels is 1(greyscale);
#We need 32 different feature maps(here, 32 feature maps means 32 filters applied on each image)
W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) #Need 32 biases for 32 outputs

#Convolve with weight tensor and add biases
#Inputs: tensor of shape, a filter/kernel, stride of kernel window
convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

#Apply the ReLU Activation Function
h_conv1 = tf.nn.relu(convolve1)

#Apply the max pooling function(Takes the max of a matrix)
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2

#Convolutional Layer 2
#The input channels is 32, because we had 32 feature maps
#64 output feature maps
w_conv2 = tf.Variable(tf.truncated_normal([5,5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #Need 64 biases for 64 outputs

#Convolve image with weight tensors and add biases
convolve2 = tf.nn.conv2d(conv1, w_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2

#Apply the ReLu activation function
h_conv2 = tf.nn.relu(convolve2)

#Apply the max pooling
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2

layer2_matrix = tf.reshape(conv2, [-1, 7*7*64])

W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 512], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[512])) # need 1024 biases for 1024 outputs

fcl=tf.matmul(layer2_matrix, W_fc1) + b_fc1

h_fc1 = tf.nn.relu(fcl)

keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([512, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]

fc=tf.matmul(layer_drop, W_fc2) + b_fc2

y_CNN= tf.nn.softmax(fc)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_CNN,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("Test Accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
print("--- %s minutes ---" % ((time.time() - start_time)/60))