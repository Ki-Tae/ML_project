import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Activation, Reshape


# 1. dataset 준비하기 >> excel 파일에서 data 불러오는 것 필요

# D**(-1) * A * X 가 input  
# training set / validation set / test set 나누기 >> 비율 어떻게 할 지?
# 이후 라벨링
A_hat = "D**(-1) * A 부분 data file에서 정의 해놓기"

# 2. 모델 구성
input_layer = Input(shape=(29,29))

model = MyModel(inputs = input_layer, outputs = E_pred)
model.compile(optimizer='rmsprop',
              loss='mean_squared_error'
              )
model.fit(labels)


class MyModel(tf.keras.Model):

  def __init__(self, num_classes=29):
    super(MyModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    # Define your layers here.
    self.layer1 = layers.Dense(29, activation='sigmoid')
    self.layer2 = layers.Dense(num_classes, activation='sigmoid')
    self.layer3 = layers.Dense(num_classes, activation='sigmoid')
    self.layer_sum = layers.Dense(1, activation='sigmoid')

  def call(self, inputs):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    x = self.layer1(inputs)
    x = A_hat*x
    x = self.layer2(x)
    x = A_hat*x
    x = self.layer3(x)

    x = self.layer_sum(X)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        E=sess.run(x)
        E_sum = 0
        for i in range(shape):
            E_sum += E[i]
    E_pred = E_sum
    return E_pred

  def compute_output_shape(self, input_shape):
    # You need to override this function if you want to use the subclassed model
    # as part of a functional-style model.
    # Otherwise, this method is optional.
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.num_classes
    return tf.TensorShape(shape)


