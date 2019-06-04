import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv

#다변인 선형회귀 모델에 영향을 미치는 변인이 여러 개 일 때 사용하는 모델
model = tf.global_variables_initializer();

data = read_csv('cancer3.csv', sep=',')
xy = np.array(data, dtype=np.float32)

x_data = xy[:,:-1]
y_data = xy[:,[-1]] #status 값

X = tf.placeholder(tf.float32, shape=[None,7])
Y = tf.placeholder(tf.float32, shape=[None,1])
W = tf.Variable(tf.random_normal([7,1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100001):
    cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 1000 == 0:
        print("#", step, "손실 비용: ", cost_)
        print("status: ", hypo_[0])

saver = tf.train.Saver()
save_path = saver.save(sess, "./saved.cpkt3")
print('학습된 모델을 저장했습니다.')
