import tensorflow as tf
import pickle
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Activation, Reshape

# 1. dataset 준비하기 >> excel 파일에서 data 불러오는 것 필요
X_dict_file_path = "C:\KT_project\dataset\A_subsets\AnX_dict_subset0.pickle"
with open(X_dict_file_path, 'rb') as f:
    AnX_dict_subset = pickle.load(f)
A = AnX_dict_subset['molecule0'][0]
X1 = AnX_dict_subset['molecule0'][1]
A_tf = tf.convert_to_tensor(A, dtype=np.float64)
X_tf = tf.convert_to_tensor(X1, dtype=np.float64)


# D**(-1) * A * X 가 input  
# training set / validation set / test set 나누기 >> 비율 어떻게 할 지?
# E_true = d['molecule energy']
E_true = 3

# 이후 라벨링
# A_hat = "D**(-1) * A 부분 data file에서 정의 해놓기"
A_hat = A_tf

def my_model(E_true, E_pred, inputs, learning_rate, iteration):
    # Loss / Optimizer
    loss = tf.losses.mean_squared_error(labels = E_true, predictions = E_pred)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(loss=loss)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iteration):
            _, loss_value = sess.run((train, loss))
            print(loss_value)
            
        
        print(sess.run(E_pred, {x : inputs}))
        print()


def get_sum_of_x(x, shape):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        E=sess.run(x)
        E_sum = 0
        for i in range(shape):
            E_sum += E[i]
    return E_sum 

# 2. layer 구성
x = tf.placeholder(tf.float64, shape=[29,29])

# 1st layer
x = tf.multiply(A_hat, x)
layer1 = tf.layers.Dense(units=29, activation='relu')
x = layer1(x)

# 2nd layer
x = tf.multiply(A_hat, x)
layer2 = tf.layers.Dense(units=29, activation='relu')
x = layer2(x)

# 3rd layer
x = tf.math.multiply(A_hat, x)
layer3 = tf.layers.Dense(units=29, activation='relu')
x = layer3(x)

# layer that combines each atom
layer_sum = tf.layers.Dense(units=1, activation='relu')
x = layer_sum(x)

# combining all 
E_pred = get_sum_of_x(x, 29)

my_model(E_true = E_true, E_pred = E_pred, inputs= X_tf, learning_rate = 0.01, iteration = 50)


