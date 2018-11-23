import numpy as np
import tensorflow as tf
import keras
from model_data import load_dict, convert_to_inputs_outputs
from random import randint

max_epoch = 20000

"""
Define input/output.
"""
size_input = 29
num_molecules = 1000
AXE_file_path = "C:\KT_project\dataset\AXE_dict_subset\AXE_dict_subset0.pickle"
AXE_dict_subset = load_dict(AXE_file_path = AXE_file_path)
results = convert_to_inputs_outputs(AXE_dict_subset=AXE_dict_subset, molecule_num=num_molecules, subset_num=0)
A_array_list = results['A']
X_array_list = results['X']
output_E_list = results['E']
print("inputs and outputs are converted.")
print(A_array_list[0].shape)

"""
Define model.
In tensorflow/keras, dimension of input matrix is [# of samples(atoms), # of features]
Thus, we need to transpose the dummy_input.
"""
inputs = tf.keras.Input(shape=(size_input, size_input), name="Input_X_layer")
A = tf.keras.Input(shape=(size_input, size_input), name="Input_A_layer")
print(A.shape)
print(inputs.shape)
outputs = tf.placeholder(tf.float32, [None, 1], name="Output_layer")
print(outputs.shape)
# 1st layer
x = tf.matmul(A, inputs, name="op_1")
x = tf.keras.layers.Dense(30, activation='relu', kernel_initializer=tf.initializers.he_normal(seed=1), name="layer1")(x)
# 2nd layer
x = tf.matmul(A, x, name="op_2")
x = tf.keras.layers.Dense(60, activation='relu', kernel_initializer=tf.initializers.he_normal(seed=1), name="layer2")(x)
# 3rd layer
x = tf.matmul(A, x, name="op_3")
x = tf.keras.layers.Dense(30, activation='relu', kernel_initializer=tf.initializers.he_normal(seed=1), name="layer3")(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.initializers.variance_scaling(scale=1.0), name="layer4")(x)
print(predictions)

"""
The dimension of predictions is [size_input*num_molecules, 1].
To sum the atomic values in same molecule, we reshape the predictions matrix and sum according to axis 0
Thus, dim [size_input*num_molecules, 1] -> (reshape)
      dim [size_input, num_molecules, 1] -> (reduce_sum)
      dim [num_molecules, 1]
"""
predictions = tf.reduce_sum(tf.reshape(predictions, [size_input, -1, 1]), axis=0, name="op_sum")
print(predictions)

"""
After defining the neural network output,
we use tensorflow to set loss function and train the model.
"""

loss = tf.losses.mean_squared_error(labels = outputs, predictions = predictions)
tf.summary.scalar('loss', loss)
with tf.name_scope("train"):
    optim = tf.train.AdamOptimizer(learning_rate=0.00001)
    #optim = tf.train.RMSPropOptimizer(learning_rate=0.01)
    #optim = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    minimize = optim.minimize(loss)


# To start tensorflow, we open the tensorflow session
with tf.Session() as sess:
    # before training, we need to initialize the variables(=weights)
    sess.run(tf.global_variables_initializer())
    merged_summary=tf.summary.merge_all()
    writer = tf.summary.FileWriter("C:\KT_project\gcn\model\hist\\6")
    writer.add_graph(sess.graph)
    # code above writes the graph in tensorboard
    for epoch in range(max_epoch):
        ran_num = randint(0,num_molecules-1)
        A_matrix = A_array_list[ran_num]
        X_matrix = X_array_list[ran_num]
        E = output_E_list[ran_num]
        minimize.run(feed_dict={inputs: X_matrix, A: A_matrix, outputs: E})
        if (epoch+1) % 100 == 0:
            # show loss for every 100th epoch
            # print(epoch+1, sess.run(loss, feed_dict={inputs:np.transpose(dummy_input), outputs:dummy_output.reshape([-1,1])}))
            # print(epoch+1, sess.run(loss, feed_dict={inputs: X_matrix, A: A_matrix, outputs: E}))
            s = sess.run(merged_summary, feed_dict={inputs: X_matrix, A: A_matrix, outputs: E})
            writer.add_summary(s, epoch+1)
            # code above writes summary of the session and writes to the disk.

# model 학습과정 표시하기
