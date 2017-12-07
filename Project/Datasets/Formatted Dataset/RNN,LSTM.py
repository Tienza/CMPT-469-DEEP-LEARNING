import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time

def next_batch(num, data, labels):
    global trainImages
    global trainLabels
    idx = np.arange(0 , len(data))
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    print(np.shape(data_shuffle))
    print(np.shape(labels_shuffle))
    print(np.shape(np.array(data_shuffle)))
    print(np.shape(np.array(labels_shuffle)))

    # Remove the num from the original lists
    trainImages = trainImages[num:]
    trainLabels = trainLabels[num:]

    print("trainImages Length: " + str(len(trainImages)))
    print("trainLabels Length: " + str(len(trainLabels)))

    return np.array(data_shuffle), np.array(labels_shuffle)

df = pd.read_csv('targets.csv', header = None)

'''for file in df[0]:
    np_arr = np.load('./ds/' + file)
    print(np_arr)

for file in df[0]:
    np_arr = np.load('./ds/r-' + file)
    print(np_arr)'''

trainImages = np.zeros((24425, 200, 491))
trainLabels = np.zeros((24425, 1))

for file in range(len(df[0])):
    np_arr = np.load('./ds/' + df[0][file])
    target = df[1][file]
    trainImages[file,:] = np_arr
    trainLabels[file] = target

# For generating psuedo random test data
rand_index = np.random.permutation(len(df[1]))
testImages = trainImages[rand_index[20000:],:]
testLabels = trainLabels[rand_index[20000:]]
trainImages = trainImages[rand_index[0:20000],:]
trainLabels = trainLabels[rand_index[0:20000]]

ntrain = trainImages.shape[0]
ntest = trainImages.shape[0]
dim = trainImages.shape[1]
nclasses = trainLabels.shape[1]
print ("Train Images: ", trainImages.shape)
print ("Train Labels  ", trainLabels.shape)
print ("Test Images:  " , testImages.shape)
print ("Test Labels:  ", testLabels.shape)

n_input = 200 # MNIST data input (img shape: 28*28)
n_steps = 491 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 1 # MNIST total classes (0-9 digits)


learning_rate = 0.001
training_iters = 19975
batch_size = 25
display_step = 1

#batch_x, batch_y = next_batch(batch_size, trainImages, trainLabels)
#exit()

# Placeholder for batches
# batch_placeholder = np.zeros((batch_size, 3600))

x = tf.placeholder(dtype="float", shape=[None, n_steps, n_input], name="x") # Current data input shape: (batch_size, n_steps, n_input) [100x28x28]
y = tf.placeholder(dtype="float", shape=[None, n_classes], name="y")

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs = x, dtype = tf.float32)

output = tf.reshape(tf.split(outputs, n_steps, axis = 1, num = None, name = 'split')[-1], [-1, n_hidden])
pred = tf.matmul(output, weights['out']) + biases['out']

cost = tf.reduce_mean(tf.losses.mean_squared_error(
    labels = y,
    predictions = pred,
    weights = 1.0,
    scope = None,
    loss_collection = tf.GraphKeys.LOSSES,
    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = pred ))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

correct_pred = tf.metrics.mean_absolute_error(
    labels = y,
    predictions = pred,
    weights = None,
    metrics_collections = None,
    updates_collections = None,
    name = None
)

'''accuracy = tf.reduce_mean(tf.metrics.mean_absolute_error(
    labels = y,
    predictions = pred))'''
#correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:

        # We will read a batch of 100 images [100 x 784] as batch_x
        # batch_y is a matrix of [100x10]
        batch_x, batch_y = next_batch(batch_size, trainImages, trainLabels)

        # We consider each row of the image as one sequence
        # Reshape data to get 28 seq of 28 elements, so that, batxh_x is [100x28x28]
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step % display_step == 0:
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("**************************************************************")
            print("Iter " + str(step * batch_size) + ", Minibatch Loss = " + \
                  "{:.6f}".format(loss))
            print("**************************************************************")
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for the testing data
    # test_len = len(rand_index)
    test_data = testImages.reshape((-1, n_steps, n_input))
    test_label = testLabels
    # Calculate batch loss
    loss = sess.run(cost, feed_dict={x: test_data, y: test_label})
    print("**************************************************************")
    print("Iter " + str(step * batch_size) + ", Testing Batch Loss = " + \
          "{:.6f}".format(loss))
    print("**************************************************************")

    tarY, predY = sess.run(y, pred, feed_dict={x: test_data, y: test_label})
    print(tarY)
    print(predY)

sess.close()
