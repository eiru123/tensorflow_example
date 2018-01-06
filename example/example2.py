import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)  # 2번째 파라미터로 데이터 타입을 줄 수 있다
node2 = tf.constant(4.0)  # 자동적으로 데이터타입이 tf.float32로 정해진다
node3 = tf.add(node1, node2)

print("node1: ", node1, "node2: ", node2)
print("node3: ", node3)

sess = tf.Session()
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))