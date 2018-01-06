import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights, try to -3.0
W = tf.Variable(5.0)

# Our hypothesis for linear model X * W
hypothesis = X * W

# Manual gradient 실제 우리가 한 gradient 와 tensorflow 의 값을 비교하기 위해 사용
gradient = tf.reduce_mean((W * X - Y) * X) * 2
# cost/Loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# gradient 를 임의로 바꾸고 싶다 할 때 사용
# gradient 를 cost 에 맞게 값을 계산해준다
# Get gradients
gvs = optimizer.compute_gradients(cost)
# Apply gradients -> gradient 를 적용하는 것 minimize 하는것
apply_gradients = optimizer.apply_gradients(gvs)

# Launch teh graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)
