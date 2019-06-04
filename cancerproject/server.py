# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

import datetime
import tensorflow as tf
import numpy as np

app = Flask(__name__)

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

# 세션 객체를 생성합니다.
sess = tf.Session()
sess.run(model)
sess2 = tf.Session()
sess2.run(model)



# 저장된 모델을 세션에 적용합니다.
save_path = "./model/saved.cpkt"
saver.restore(sess, save_path)
save_path2 = "./model/saved.cpkt2"
saver.restore(sess2, save_path2)
save_path3 = "./model/saved.cpkt3"
saver.restore(sess3, save_path3)
save_path4 = "./model/saved.cpkt4"
saver.restore(sess4, save_path4)
save_path5 = "./model/saved.cpkt"
saver.restore(sess5, save_path5)
save_path6 = "./model/saved.cpkt6"
saver.restore(sess6, save_path6)
save_path9 = "./model/saved.cpkt9"
saver.restore(sess9, save_path)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # 파라미터를 전달 받습니다.
        age = float(request.form['age'])
        family_history = float(request.form['family_history'])
        exercise = float(request.form['exercise'])
        alcohol = float(request.form['alcohol'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        smoking = float(request.form['smoking'])
        sex = float(request.form['sex'])
        name = str(request.form['name'])

        #bmi 계산
        bmi_group = (weight)/((height*0.01)*(height*0.01))

        if bmi_group>=10 and bmi_group<=24.99:
            bmi_group=1
        elif bmi_group>=25 and bmi_group<=29.99:
            bmi_group=2
        elif bmi_group>=30 and bmi_group<=34.99:
            bmi_group=3
        elif bmi_group>=35:
            bmi_group=4
        else:
            bmi_group=9


        # 암 발병 상태 번수를 선언합니다.
        status = 0
        status2 = 0
        status3 = 0
        status4 = 0
        status5 = 0
        status6 = 0
        status9 = 0

        # 입력된 파라미터를 배열 형태로 준비합니다.
        data = ((age, family_history, exercise, alcohol, bmi_group, smoking, sex),(0, 0, 0, 0, 0, 0, 0))
        arr = np.array(data, dtype=np.float32)

        # 입력 값을 토대로 예측 값을 찾아냅니다.
        x_data = arr[0:7]
        dict = sess.run(hypothesis, feed_dict={X: x_data})
        dict = sess2.run(hypothesis, feed_dict={X: x_data})
        dict = sess3.run(hypothesis, feed_dict={X: x_data})
        dict = sess4.run(hypothesis, feed_dict={X: x_data})
        dict = sess5.run(hypothesis, feed_dict={X: x_data})
        dict = sess6.run(hypothesis, feed_dict={X: x_data})
        dict = sess9.run(hypothesis, feed_dict={X: x_data})
            
        # 암 발병률 결과를 저장합니다.
        status = dict[0]
        status2 = dict[1]
        status3 = dict[2]
        status4 = dict[3]
        status5 = dict[4]
        status6 = dict[5]
        status9 = dict[6]

        num = np.round(status,4)
        num2 = np.round(status2,4)

        return render_template('index.html', name= name, status=num, status2=num2,
         status3=status3, status4=status4, status5=status5, status6=status6, status9=status9)


if __name__ == '__main__':
   app.run(debug = True)