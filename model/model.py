import os
import numpy as np
import tensorflow as tf
import keras
from model_data import load_dict, convert_to_inputs_outputs, convert_to_muX, split_data_set
from iterator import custom_iterator
from random import randint

max_epoch = 200

change_dataset_after_nth_epoch = 100

"""
Define input/output.
"""
which_input_list = ["AXEN", "muX"]
which_output_list = ['U', 'G', 'Cv', 'H']
which_input = which_input_list[0]
which_output = which_output_list[0]
atom_num_range = range(18, 21)
test_atom_num_range = range(18, 21)

val_set_size = 1500
test_set_size = 1500


size_input = 29


"""
Define model.
In tensorflow/keras, dimension of input matrix is [# of samples(atoms), # of features]
Thus, we need to transpose the dummy_input.
"""
inputs = tf.keras.Input(shape=(size_input, size_input), name="Input_X_layer")
A = tf.keras.Input(shape=(size_input, size_input), name="Input_A_layer")
N = tf.placeholder(tf.int32, shape=(size_input,1), name = "Number_of_atom")
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
read_out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.initializers.variance_scaling(scale=1.0), name="layer4")(x)
"""
The dimension of predictions is [size_input*num_molecules, 1].
To sum the atomic values in same molecule, we reshape the predictions matrix and sum according to axis 0
Thus, dim [size_input*num_molecules, 1] -> (reshape)
      dim [size_input, num_molecules, 1] -> (reduce_sum)
      dim [num_molecules, 1]read_out
"""
# predictions = tf.reduce_sum(tf.reshape(read_out, [size_input, -1, 1]), axis=0, name="op_row_sum")
# predictions = tf.segment_sum(tf.reshape(read_out, [size_input, -1, 1]), tf.reshape(N, [size_input, -1, 1]) , name="op_sum")
seg_sum = tf.segment_sum(tf.reshape(read_out, [-1]), tf.reshape(N, [-1]), name="op_seg_sum")
predictions = tf.reshape(seg_sum[0], [-1, 1])

"""
After defining the neural network output,
we use tensorflow to set loss function and train the model.
"""

loss_adam = tf.losses.mean_squared_error(labels = outputs, predictions = predictions)
loss_adam_val = tf.losses.mean_squared_error(labels = outputs, predictions = predictions)

# tf.summary.scalar('loss_adam', loss_adam)
# tf.summary.scalar('loss_adam_validation', loss_adam_val)

with tf.name_scope("train"):
    optim1 = tf.train.AdamOptimizer(learning_rate=0.0001)
    minimize1 = optim1.minimize(loss_adam)

# saver
SAVER_DIR ="C:\KT_project\gcn\model\model_chkp\\AXEN_train_15to21\\"
saver = tf.train.Saver(max_to_keep=int(max_epoch/change_dataset_after_nth_epoch))
# instantiating Saver. This saver keeps latest n ckpts 

# To start tensorflow, we open the tensorflow session
with tf.Session() as sess:
    # before training, we need to initialize the variables(=weights)
    sess.run(tf.global_variables_initializer())
    
    ask_run_test = input("Do you want to run the test? [Y/n]")

    if ask_run_test == "Y":

        saver = tf.train.import_meta_graph('C:\KT_project\gcn\model\model_chkp\\AXEN_train_15to21\\-8900.meta')
        saver.restore(sess, save_path='C:\\KT_project\\gcn\\model\\model_chkp\\AXEN_train_15to21\\-8900')
        loss_test_list = []
        
        length_test_set_list = 0
        while length_test_set_list <= test_set_size:

            test_result= custom_iterator(which_input = which_input, which_output = which_output, atom_num_range = test_atom_num_range)
            
            for i in range(test_result['length']):
                
                A_ts_matrix = test_result['AXEN']['A'][i]
                X_ts_matrix = test_result['AXEN']['X'][i]
                N_ts_matrix = test_result['AXEN']['N'][i]
                if which_output == 'U':
                    output_ts = test_result['AXEN']['E'][i]
                elif which_output != 'U':
                    output_ts = test_result['output'][i]
                
                loss_test = sess.run(loss_adam_val, feed_dict={inputs: X_ts_matrix, A: A_ts_matrix, N: N_ts_matrix, outputs: output_ts})
                prediction_value = sess.run(predictions, feed_dict={inputs: X_ts_matrix, A: A_ts_matrix, N: N_ts_matrix, outputs: output_ts})
                output_value = output_ts[0,0]
                
                print(i, prediction_value)
                print(i, output_value)
                loss_test_list.append(loss_test)
            
            length_test_set_list += test_result['length']
            
        loss_test_value = sum(loss_test_list)/len(loss_test_list)
        print("Loss for train set data is {}".format(loss_test_value))
        
    # merged_summary=tf.summary.merge_all()
    # merged_summary=tf.summary.merge(['loss_train', 'loss_valid'])
    # writer = tf.summary.FileWriter("C:\KT_project\gcn\model\hist\\test\\result1")
    # writer.add_graph(sess.graph)
    # code above writes the graph in tensorboard

    ask_run_train = input("Train? [Y/n]")
    if ask_run_train == "Y":    
        for n in range(int(max_epoch/change_dataset_after_nth_epoch)):
            # designate dataset
            result= custom_iterator(which_input = which_input, which_output = which_output, atom_num_range = atom_num_range)
            val_result = custom_iterator(which_input = which_input, which_output = which_output, atom_num_range = atom_num_range)
            
            for epoch in range(change_dataset_after_nth_epoch):              
                random_num = np.random.randint(0, result['length'])
                A_tr_matrix = result['AXEN']['A'][random_num]
                X_tr_matrix = result['AXEN']['X'][random_num]
                N_tr_matrix = result['AXEN']['N'][random_num]
                if which_output == 'U':
                    output_tr = result['AXEN']['E'][random_num]
                elif which_output != 'U':
                    output_tr = result['output'][random_num]
                minimize1.run(feed_dict={inputs: X_tr_matrix, A: A_tr_matrix, N: N_tr_matrix, outputs: output_tr})
                # train only occurs here :D
               
                if n*change_dataset_after_nth_epoch + epoch == 0:
                    # print out initial loss value
                    print(epoch+1, sess.run(loss_adam, feed_dict={inputs: X_tr_matrix, A: A_tr_matrix, N: N_tr_matrix, outputs: output_tr}))
                    # print(epoch+1, sess.run(read_out, feed_dict={inputs: X_tr_matrix, A: A_tr_matrix, N: N_tr_matrix, outputs: output_tr}))
                    val_random_num = np.random.randint(0, val_result['length'])
                    A_val_matrix = val_result['AXEN']['A'][val_random_num]
                    X_val_matrix = val_result['AXEN']['X'][val_random_num]
                    N_val_matrix = val_result['AXEN']['N'][val_random_num]
                    if which_output == 'U':
                        output_val = val_result['AXEN']['E'][val_random_num]
                    elif which_output != 'U':
                        output_val = val_result['output'][val_random_num]
                    print(epoch+1, sess.run(loss_adam_val, feed_dict={inputs: X_val_matrix, A: A_val_matrix, N: N_val_matrix, outputs: output_val}))

                if (epoch+1) % 100 == 0:
                    # show loss for every 100th epoch
                    # below saves the model weights and etc.
                    saver.save(sess, save_path = SAVER_DIR, global_step = n*change_dataset_after_nth_epoch + epoch +1)
                
                    # below are train set loss and validation set loss
                    print(n*change_dataset_after_nth_epoch+epoch+1, sess.run(loss_adam, feed_dict={inputs: X_tr_matrix, A: A_tr_matrix, N: N_tr_matrix, outputs: output_tr}))
                    # print(n*change_dataset_after_nth_epoch+epoch+1, sess.run(read_out, feed_dict={inputs: X_tr_matrix, A: A_tr_matrix, N: N_tr_matrix, outputs: output_tr}))
                    # calculate validation_set_loss
                    print(n*change_dataset_after_nth_epoch+epoch+1, sess.run(read_out, feed_dict={inputs: X_tr_matrix, A: A_tr_matrix, N: N_tr_matrix, outputs: output_tr}))
                    print(n*change_dataset_after_nth_epoch+epoch+1, sess.run(seg_sum, feed_dict={inputs: X_tr_matrix, A: A_tr_matrix, N: N_tr_matrix, outputs: output_tr}))
                    loss_adam_val_list = []
                    for val_num in range(val_result['length']):
                        A_val_matrix = val_result['AXEN']['A'][val_num]
                        X_val_matrix = val_result['AXEN']['X'][val_num]
                        N_val_matrix = val_result['AXEN']['N'][val_num]
                        if which_output == 'U':
                            output_val = val_result['AXEN']['E'][val_num]
                        elif which_output != 'U':
                            output_val = val_result['output'][val_num]
                        loss_val = sess.run(loss_adam_val, feed_dict={inputs: X_val_matrix, A: A_val_matrix, N: N_val_matrix, outputs: output_val})
                        loss_adam_val_list.append(loss_val)
                    loss_adam_val_value = sum(loss_adam_val_list)/len(loss_adam_val_list)
                    print(n*change_dataset_after_nth_epoch+epoch+1, loss_adam_val_value)

                    """
                    s_train = sess.run(merged_summary, feed_dict={inputs: X_tr_matrix, A: A_tr_matrix, outputs: E_tr})
                    s_valid = sess.run(loss_adam_val_value)
                    # s = sess.run(merged_summary, feed_dict={inputs: muX_matrix, A: A_matrix, outputs: E})
                    
                    # write summary into tensorboard
                    writer.add_summary(s_train, epoch+1)
                    writer.add_summary(s_valid, epoch+1)
                    """