from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()

model.add(Dense(32, input_shape = (29, 29)))
model.add(Activation('relu'))

