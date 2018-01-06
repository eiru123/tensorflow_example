import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])
            # img 28 * 28 * 1 (black/white)
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # L1 ImgIn shape=(?, 28, 28, 1)
            # L1 ImgIn shape=(?, 28, 28, 1)
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            # Conv -> (?, 28, 28, 32)
            # Pool -> (?, 14, 14, 32)
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

            # L1 ImgIn shape=(?, 28, 28, 1)
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            # Conv -> (?, 28, 28, 32)
            # Pool -> (?, 14, 14, 32)
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.reshape(L2, [-1, 7 * 7 * 64])  # 겹쳐져 있는 층을 다 길게 나열한다.

            # Final FC 7 * 7 * 64 inputs -> 10 outputs
            W3 = tf.get_variable("W2", shape=[7 * 7 * 64, 10],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([10]))
            hypothesis = tf.matmul(L2, W3) + b

            # define cost/loss & optimizer
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop= 1.0):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop = 0.7):
        return self.sess.run([self.cost, self.optimizer],
                             feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")
sess.run(tf.global_variables_initializer())

# epoch 전체 데이터 셋을 한 번 트레이닝 돌린 것. 근데 너무 많으면 데이터를 짤라서 해야하기 때문에 이 때 나눈 것들을 사용한 것.
training_epochs = 15
batch_size = 100
# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch: ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

print('Learning Finished!')



# layers 찾아보
