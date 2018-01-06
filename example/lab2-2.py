import tensorflow as tf

# building graph using TF operations

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# tf.placeholder 사용
# shape 도 넘겨줄 수 있다.
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# trainable 한 것. 텐서플로를 실행시켰을때 텐서플로가 스스로 값을 변경시키는 값
# random_normal(shape)
W = tf.Variable(tf.random_normal([1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))

# Our hypothesis WX+b
hypothesis = X * W + b

# cost/loss function
# reduce_mean 은 시그마를 표현 [1., 2., 3., 4.] tf.reduce_mean(t) -> 2.5
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# W, b를 변경해서 스스로 cost 가 가장 적은 W, b를 구한다
train = optimizer.minimize(cost)

# launch the graph in a session.
sess = tf.Session()

# Initializes global variables in the graph.
# tf.variable 을 사용하기 위해서는 꼭 tf.global_variables_initializer()를 실행시켜야 한다
sess.run(tf.global_variables_initializer())

# placeholder 를 사용하는 이유는 우리의 모델에 값들을 따로 지정할 수 있다.
# W = 1, b = 0 에 근사해야 좋은 모델
# Fit the line
for step in range(2001):
    # 리스트에 넣어서 모두 한 번에 실행시킬 수 있다.
    cost_val, W_val, b_val, _ = \
    sess.run([cost, W, b, train],
             feed_dict={X: x_train, Y: y_train})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

