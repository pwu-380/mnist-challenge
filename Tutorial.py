#Import training data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # This is our training data

import tensorflow as tf
sess = tf.InteractiveSession()
saver = tf.train.Saver()

# Note: the following are placeholders for a tensor that will be constantly fed
x = tf.placeholder(tf.float32, shape=[None, 784]) # Feature tensor = 28x28
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # Class actual (0..9) 1-hot encoded

W = tf.Variable(tf.zeros([784,10])) # Weights
b = tf.Variable(tf.zeros([10]))     # Biases

sess.run(tf.initialize_all_variables())

# Model
y = tf.matmul(x,W) + b

# Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

# Calculates next step by gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# Pretty cool that tensorflow updates the parameters for you, very streamlined

#Training
for i in range(1000):
    batch = mnist.train.next_batch(100)                     # Batch size = 100
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})   # Trains model
    # Note: feed_dict matches the placeholder with the attribute tensors

#Model scoring
correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#Model saving
save_path = saver.save(sess, "linear_model.ckpt")
print ("Model saved as: ", save_path)