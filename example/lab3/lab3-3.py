import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights, try to -3.0
W = tf.Variable(5.0)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/Loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Launch teh graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)
