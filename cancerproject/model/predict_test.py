import tensorflow as tf
import numpy as np

# 플레이스 홀더를 설정합니다.
X = tf.placeholder(tf.float32, shape=[None, 7])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([7, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# 가설을 설정합니다.
hypothesis = tf.matmul(X, W) + b

# 저장된 모델을 불러오는 객체를 선언합니다.
saver = tf.train.Saver()
model = tf.global_variables_initializer()


# 7가지 변수를 입력 받습니다.
age = float(input('연령: '))
family_history = float(input('가족력: '))
exercise = float(input('운동: '))
alcohol = float(input('음주여부: '))
bmi_group = float(input('bmi: '))
smoking = float(input('흡연여부: '))
sex = float(input('성별: '))

with tf.Session() as sess:
    sess.run(model)
    save_path = "./saved.cpkt5"
#    save_path2 = "./saved.cpkt2"
    saver.restore(sess, save_path)
#    saver.restore(sess, save_path2)

    data = ((age, family_history, exercise, alcohol, bmi_group, smoking, sex), )
    arr = np.array(data, dtype=np.float32)

    x_data = arr[0:7]
    dict = sess.run(hypothesis, feed_dict={X: x_data})

    print(dict[0])
