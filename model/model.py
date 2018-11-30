import os
import numpy as np
import tensorflow as tf
import keras
from model_data import load_dict, convert_to_inputs_outputs, convert_to_muX, split_data_set, train_saver
from random import randint

max_epoch = 3000

"""
Define input/output.
"""

size_input = 29
num_molecules = 1000
train_set_size = 600
valid_set_size = 200
test_set_size = 50

AXEN_file_path = "C:\KT_project\dataset\AXEN_dict_subset\AXEN_dict_subset0.pickle"
AXEN_dict_subset = load_dict(file_path = AXEN_file_path)

# muX_file_path = "C:\KT_project\dataset\muX_subsets\muX_dict_subset0.pickle"
# muX_dict_subset = load_dict(file_path = muX_file_path)

results = convert_to_inputs_outputs(AXEN_dict_subset=AXEN_dict_subset, molecule_num=num_molecules, subset_num=0)
# muX_array_list = convert_to_muX(AXEN_dict_subset=AXEN_dict_subset, muX_dict_subset=muX_dict_subset, molecule_num=num_molecules, subset_num=0)
split_results = split_data_set(result=results, train_set_size=train_set_size, valid_set_size=valid_set_size, test_set_size=test_set_size)

# designating train/validation dataset
A_train = split_results['A_train']
X_train = split_results['X_train']
E_train = split_results['E_train']

A_valid = split_results['A_valid']
X_valid = split_results['X_valid']
E_valid = split_results['E_valid']

A_test = split_results['A_test']
X_test = split_results['X_test']
E_test = split_results['E_test']
# A_array_list = results['A']
# X_array_list = results['X']
# output_E_list = results['E']
# atom_num_list = results['N']

print("inputs and outputs are loaded.")

"""
Define model.
In tensorflow/keras, dimension of input matrix is [# of samples(atoms), # of features]
Thus, we need to transpose the dummy_input.
"""
inputs = tf.keras.Input(shape=(size_input, size_input), name="Input_X_layer")
A = tf.keras.Input(shape=(size_input, size_input), name="Input_A_layer")
outputs = tf.placeholder(tf.float32, [None, 1], name="Output_layer")
# 1st layer
x = tf.matmul(A, inputs, name="op_1")
x = tf.keras.layers.Dense(50, activation='relu', kernel_initializer=tf.initializers.he_normal(seed=1), name="layer1")(x)

# 2nd layer
x = tf.matmul(A, x, name="op_2")
x = tf.keras.layers.Dense(100, activation='relu', kernel_initializer=tf.initializers.he_normal(seed=1), name="layer2")(x)
# 3rd layer
x = tf.matmul(A, x, name="op_3")
x = tf.keras.layers.Dense(50, activation='relu', kernel_initializer=tf.initializers.he_normal(seed=1), name="layer3")(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.initializers.variance_scaling(scale=1.0), name="layer4")(x)
"""
The dimension of predictions is [size_input*num_molecules, 1].
To sum the atomic values in same molecule, we reshape the predictions matrix and sum according to axis 0
Thus, dim [size_input*num_molecules, 1] -> (reshape)
      dim [size_input, num_molecules, 1] -> (reduce_sum)
      dim [num_molecules, 1]
"""
predictions = tf.reduce_sum(tf.reshape(predictions, [size_input, -1, 1]), axis=0, name="op_row_sum")
# predictions = tf.segment_sum(predictions, name="op_sum")

"""
After defining the neural network output,
we use tensorflow to set loss function and train the model.
"""

loss_adam = tf.losses.mean_squared_error(labels = outputs, predictions = predictions)
loss_adam_val = tf.losses.mean_squared_error(labels = outputs, predictions = predictions)

tf.summary.scalar('loss_adam', loss_adam)
tf.summary.scalar('loss_adam_validation', loss_adam_val)

with tf.name_scope("train"):
    optim1 = tf.train.AdamOptimizer(learning_rate=0.0001)
    minimize1 = optim1.minimize(loss_adam)

# train the model only with the train data
# with tf.name_scope("val"):
#     optim1 = tf.train.AdamOptimizer(learning_rate=0.0001)
#     minimize1 = optim1.minimize(loss_adam)

# saver
"""
SAVER_DIR = "model"
saver = tf.train.Saver(max_to_keep=3)
checkpoint_path = os.path.join(SAVER_DIR, "model")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)
"""
SAVER_DIR ="C:\KT_project\gcn\model\model_chkp\\"
# saver = train_saver()
saver = tf.train.Saver(max_to_keep=2)
# this saver keeps latest 30 ckpts 
# ckpt = tf.train.get_checkpoint_state(SAVER_DIR, latest_filename="model")


# To start tensorflow, we open the tensorflow session
with tf.Session() as sess:
    # before training, we need to initialize the variables(=weights)
    sess.run(tf.global_variables_initializer())
    
    ask_run_test = input("Do you want to run the test? [Y/n]")

    if ask_run_test == "Y":

        saver = tf.train.import_meta_graph('C:\KT_project\gcn\model\model_chkp\-3000.meta')
        saver.restore(sess, save_path='C:\\KT_project\\gcn\\model\\model_chkp\\-3000')
        loss_test_list = []
        for i in range(test_set_size):
            X_ts_matrix = X_test[i]
            A_ts_matrix = A_test[i]
            E_ts = E_test[i]
            loss_test = sess.run(loss_adam_val, feed_dict={inputs: X_ts_matrix, A: A_ts_matrix, outputs: E_ts})
            loss_test_list.append(loss_test)

        loss_test_value = sum(loss_test_list)/len(loss_test_list)
        print("Loss for train set data is {}".format(loss_test_value))

    # merged_summary=tf.summary.merge_all()
    # writer = tf.summary.FileWriter("C:\KT_project\gcn\model\hist\\with_validation")
    # writer.add_graph(sess.graph)
    # code above writes the graph in tensorboard

    ask_run_train = input("Train? [Y/n]")
    if ask_run_train == "Y":    
        for epoch in range(max_epoch):
            ran_num = randint(0,train_set_size-1)
            A_tr_matrix = A_train[ran_num]
            X_tr_matrix = X_train[ran_num]
            E_tr = E_train[ran_num]
            # muX_matrix = muX_array_list[ran_num]
            minimize1.run(feed_dict={inputs: X_tr_matrix, A: A_tr_matrix, outputs: E_tr})
            # train only occurs here :D
            
            if epoch == 0:
                print(epoch+1, sess.run(loss_adam, feed_dict={inputs: X_tr_matrix, A: A_tr_matrix, outputs: E_tr}))
                val_ran_num = randint(0,valid_set_size-1)
                A_val_matrix = A_valid[val_ran_num]
                X_val_matrix = X_valid[val_ran_num]
                E_val = E_valid[val_ran_num]
                print(epoch+1, sess.run(loss_adam_val, feed_dict={inputs: X_val_matrix, A: A_val_matrix, outputs: E_val}))
                
                # print(epoch+1, sess.run(loss_adam, feed_dict={inputs: muX_matrix, A: A_matrix, outputs: E}))

            if (epoch+1) % 100 == 0:
                # show loss for every 100th epoch
                # below saves the model weights and etc.
                saver.save(sess, save_path = SAVER_DIR, global_step = epoch + 1)
                print(epoch+1, sess.run(loss_adam, feed_dict={inputs: X_tr_matrix, A: A_tr_matrix, outputs: E_tr}))
                # below process is for achieving the validation data set loss.
                loss_adam_val_list = []
                for val_num in range(valid_set_size):
                    A_val_matrix = A_valid[val_num]
                    X_val_matrix = X_valid[val_num]
                    E_val = E_valid[val_num]
                    loss_val = sess.run(loss_adam_val, feed_dict={inputs: X_val_matrix, A: A_val_matrix, outputs: E_val})
                    loss_adam_val_list.append(loss_val)
                loss_adam_val_value = sum(loss_adam_val_list)/len(loss_adam_val_list)
                print(epoch+1, loss_adam_val_value)
                
                # s_train = sess.run(merged_summary, feed_dict={inputs: X_tr_matrix, A: A_tr_matrix, outputs: E_tr})
                # s_valid = sess.run(merged_summary, feed_dict={inputs: X_val_matrix, A: A_val_matrix, outputs: E_val})
                # s = sess.run(merged_summary, feed_dict={inputs: muX_matrix, A: A_matrix, outputs: E})
                
                # write summary into tensorboard
                # writer.add_summary(s_train, epoch+1)
                # writer.add_summary(s_valid, epoch+1)
        