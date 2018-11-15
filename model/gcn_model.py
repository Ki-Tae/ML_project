from keras.models import Model
from keras.layers import Input, Dense, Activation, Reshape

# 1. dataset 준비하기 >> excel 파일에서 data 불러오는 것 필요

# D**(-1) * A * X 가 input  
# training set / validation set / test set 나누기 >> 비율 어떻게 할 지?
# 이후 라벨링
A_hat = "D**(-1) * A 부분 data file에서 정의 해놓기"

# 2. 모델 구성
input_layer = Input(shape=(29,29))

# 1st layer
x = Dense(29, activation = 'relu')(input_layer)

# 2nd layer
x = A_hat*x
x = Dense(29, activation = 'relu')(x)

# 3rd layer
x = A_hat*x
x = Dense(29, activation = 'relu')(x)

# last prediction should sum all the data 
prediction = Dense(1, activation = 'relu')(x)


model = Model(inputs = input_layer, outputs = target_output)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',  """loss function > mse로"""
              metrics=['accuracy'])
model.fit(labels)


"""
model = Sequential()
# 1st layer
model.add(Dense(29, input_shape = (29, 29)))
# input shape 이 29*29 matrix이므로 다음과 같이 정의
# outpur shape은 마찬가지로 29*29이어야 하므로 unit 자리에 29
model.add(Activation('relu'))

# 나온 output에 A를 곱해줘야 되는데 Dense function으로 가능?
# 2nd and 3rd layer
model.add(Dense(29))
model.add(Activation('relu'))

model.add(Dense(29))
model.add(Activation('relu'))
# 마지막 layer는 29 * 1 vector이어야 하고, 이를 마지막에 다 더한 값이 molecule의 energy가 되어야함

model.add(Reshape(1,1))
"""


