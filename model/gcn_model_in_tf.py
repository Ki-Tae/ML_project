import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Activation, Reshape

# 1. dataset 준비하기 >> excel 파일에서 data 불러오는 것 필요

# D**(-1) * A * X 가 input  
# training set / validation set / test set 나누기 >> 비율 어떻게 할 지?
E_true = d['molecule energy']/ 
# 이후 라벨링
A_hat = "D**(-1) * A 부분 data file에서 정의 해놓기"

# 2. layer 구성
x = tf.placeholder(tf.float64, shape=[29,29])

# 1st layer
x = A_hat*x
layer1 = tf.layers.Dense(units=29, activation='relu')
x = layer1(x)

# 2nd layer
x = A_hat*x
layer2 = tf.layers.Dense(units=29, activation='relu')
x = layer2(x)

# 3rd layer
x = A_hat*x
layer3 = tf.layers.Dense(units=29, activation='relu')
x = layer3(x)

# layer that combines each atom
layer_sum = tf.layers.Dense(units=1, activation='relu')
x = layer_sum(x)

# combining all 
E_pred = get_sum_of_x(x, 29)

# Loss
loss = tf.losses.mean_squared_error(labels = E_)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)



def get_sum_of_x(x, shape):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        E=sess.run(x)
        E_sum = 0
        for i in range(shape):
            E_sum += E[i]
    return E_sum 
