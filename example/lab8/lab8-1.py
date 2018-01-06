import numpy as np
import pprint as pp
import tensorflow as tf
t = np.array([0.1, 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print(t.ndim)
print(t.shape)

sess = tf.Session()
t = tf.constant([1, 2, 3, 4])
print(sess.run(tf.shape(t)))

t = tf.constant([[1, 2],
                 [3, 4]])
print(sess.run(tf.shape(t)))

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
print(sess.run(tf.shape(t)))

matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[1.], [2.]])
print("Matrix 1 shape", matrix1.shape)
print("Matrix 2 shape", matrix2.shape)
print(sess.run(tf.matmul(matrix1, matrix2)))

# Operatinos between the same shapes
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
print(sess.run((matrix1 + matrix2)))
print(sess.run(tf.expand_dims([[0, 1, 2]], 2)))
print(sess.run(tf.shape([1, 2, 3])))
