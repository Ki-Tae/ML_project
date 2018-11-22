import numpy as np
import tensorflow as tf
import keras
from model_data import load_dict, convert_to_inputs_outputs
from random import randint

max_epoch = 1000

"""
Define dummy input/output.
"""
size_input = 29
num_molecules = 100
# dummy_input = 2*np.random.random([size_input, size_input*num_molecules])-1
# dummy_output = np.random.random([num_molecules])*29
AXE_file_path = "C:\KT_project\dataset\AXE_dict_subset\AXE_dict_subset0.pickle"
AXE_dict_subset = load_dict(AXE_file_path = AXE_file_path)
input_array_list, E_outputs = convert_to_inputs_outputs(AXE_dict_subset=AXE_dict_subset, molecule_num=num_molecules, subset_num=0)
print("inputs and outputs are converted.")

"""
Define model.
In tensorflow/keras, dimension of input matrix is [# of samples(atoms), # of features]
Thus, we need to transpose the dummy_input.
"""
inputs = tf.keras.Input(shape=(size_input, size_input))
A = tf.keras.Input(shape=(size_input, size_input))
x = inputs
print(A.shape)
print(x.shape)
outputs = tf.placeholder(tf.float32, [None, 1])
print(outputs.shape)
# 1st layer
x = tf.matmul(A, x)
x = tf.keras.layers.Dense(30, activation='sigmoid', kernel_initializer=tf.initializers.truncated_normal(0.3, dtype=tf.float32))(x)
# 2nd layer
x = tf.matmul(A, x)
x = tf.keras.layers.Dense(60, activation='sigmoid')(x)
# 3rd layer
x = tf.matmul(A, x)
x = tf.keras.layers.Dense(30, activation='sigmoid')(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
print(predictions)

"""
The dimension of predictions is [size_input*num_molecules, 1].
To sum the atomic values in same molecule, we reshape the predictions matrix and sum according to axis 0
Thus, dim [size_input*num_molecules, 1] -> (reshape)
      dim [size_input, num_molecules, 1] -> (reduce_sum)
      dim [num_molecules, 1]
"""
predictions = tf.reduce_sum(tf.reshape(predictions, [size_input, -1, 1]), axis=0)
print(predictions)

"""
After defining the neural network output,
we use tensorflow to set loss function and train the model.
"""
loss = tf.losses.mean_squared_error(labels = outputs, predictions = predictions)
optim = tf.train.AdamOptimizer(learning_rate=0.01)
#optim = tf.train.RMSPropOptimizer(learning_rate=0.01)
#optim = tf.train.GradientDescentOptimizer(learning_rate=0.01)
minimize = optim.minimize(loss)


# To start tensorflow, we open the tensorflow session
with tf.Session() as sess:
    # before training, we need to initialize the variables(=weights)
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("C:\KT_project\gcn\model\hist")
    writer.add_graph(sess.graph)
    # tb_hist = keras.callbacks.TensorBoard(log_dir='C:\KT_project\gcn\model\hist', histogram_freq=0, write_graph=True, write_images=True)
    for epoch in range(max_epoch):
        # minimize.run(feed_dict={inputs:np.transpose(dummy_input), outputs:dummy_output.reshape([-1,1])})
        ran_num = randint(0,num_molecules-1)
        A_matrix = input_array_list[ran_num][0]
        print(type(A_matrix))
        print(A_matrix.shape)
        inputs = input_array_list[ran_num][1]
        print(type(inputs))
        print(inputs.shape)
        outputs = np.float32(E_outputs[ran_num])
        print(type(outputs))
        minimize.run(feed_dict={inputs: inputs, A: A_matrix, outputs: outputs})
        if (epoch+1) % 100 == 0:
            # show loss for every 100th epoch
            # print(epoch+1, sess.run(loss, feed_dict={inputs:np.transpose(dummy_input), outputs:dummy_output.reshape([-1,1])}))
            print(epoch+1, sess.run(loss, feed_dict={inputs: inputs, A: A_matrix, outputs: outputs}))

