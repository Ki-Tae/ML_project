import numpy as np
import pickle
import tensorflow as tf

# loading the data
"""
each X are normalized differently.
"""

"""
max U0 ['molecule0'] = -1101.4877900833399  = -40.478909210233205eV
min U0 ['molecule133620'] = -19444.3873485546 = -714.5676940015802eV
1 ha = 27.2114 ev
"""
# minE = 1101.4877900833399
# maxE = 19444.3873485546

minE = 40.478909210233205
maxE = 714.5676940015802

maxX = 0.5*9**2.4

min_muX = -0.5
max_muX = 0.5

"""
G : max -1102.022956412281 min -19445.314630956153
H : max -1101.384033068429 min -19444.146473365512
Cv : max 46.969 min 6.002
"""

max_G = 19445.314630956153
min_G = 1102.022956412281

max_H = 19444.146473365512
min_H = 1101.384033068429

max_Cv = 46.969
min_Cv = 6.002

max_min_dict = {
    'G' : [max_G, min_G],
    'H' : [max_H, min_H],
    'Cv' : [max_Cv, min_Cv]
}

def load_dict(file_path):
    with open(file_path, 'rb') as f:
        dict_subset = pickle.load(f)

    return dict_subset

def convert_to_inputs_outputs(AXEN_dict_subset, molecule_num_list, subset_num):
    A_array_list = []
    X_array_list = []
    output_E_list = []
    molecule_atom_num_list = []
    
    for i in molecule_num_list:
        A_hat = AXEN_dict_subset['molecule{}'.format(i)][0]
        D_hat = np.diag(np.sum(A_hat, axis=0))
        D_inv = np.linalg.inv(D_hat)
        A = np.dot(D_inv, A_hat)

        A_array = [
            A
        ]
        A_array = np.asarray(A_array)
        # A_array's shape is (1,29,29)

        num = AXEN_dict_subset['molecule{}'.format(i)][3]
        N = np.ones([1,29], dtype=np.int32)
        for N_num in range(num):
            N[0, N_num] = 0

        N_array = [
            N
        ]
        N_array = np.asarray(N_array).reshape([-1,1])
        # N_array's shape is (1,29,1)

        # X normalisation
        X = AXEN_dict_subset['molecule{}'.format(i)][1]
        # X = X/maxX/N**2
        X = X/maxX
        X_array = [
            X
        ]
        X_array = np.asarray(X_array)
        # X_array's shape is (1,29,29)

        E = AXEN_dict_subset['molecule{}'.format(i)][2]/27.2114
        # energy in ev
        E = (-E-minE)/(maxE-minE)
        # E = -1 + 2*(-E-minE)/(maxE-minE)
        # E normalisation btw (0, 1)
        E_array = [
            E
        ]
        E_array = np.asarray(E_array).reshape([-1,1])
        # E_array's shape is (1,1)

        

        A_array_list.append(A_array)
        X_array_list.append(X_array)
        output_E_list.append(E_array)
        molecule_atom_num_list.append(N_array)
        
    result = {
        'A' : A_array_list,
        'X' : X_array_list,
        'E' : output_E_list,
        'N' : molecule_atom_num_list
    }

    return  result

def convert_to_muX(AXEN_dict_subset, muX_dict_subset, molecule_num_list, subset_num):
    muX_array_list = []
    
    for i in molecule_num_list:
        
        N = AXEN_dict_subset['molecule{}'.format(subset_num*1000 + i)][3]
        
        # muX normalisation
        muX = muX_dict_subset['molecule{}'.format(subset_num*1000 + i)]
        muX = muX/N**2
        # max_muX - min_muX = 1
        muX_array = [
            muX
        ]
        muX_array = np.asarray(muX_array)
        # muX_array's shape is (1,29,29)

        muX_array_list.append(muX_array)
    
    return  muX_array_list

def split_data_set(result, train_set_size, valid_set_size, test_set_size):
    A_train = result['A'][0:train_set_size]
    A_valid = result['A'][train_set_size:train_set_size+valid_set_size]
    X_train = result['X'][0:train_set_size]
    X_valid = result['X'][train_set_size:train_set_size+valid_set_size]
    E_train = result['E'][0:train_set_size]
    E_valid = result['E'][train_set_size:train_set_size+valid_set_size]
    
    A_test = []
    X_test = []
    E_test = []
    
    for i in range(test_set_size):
        A_test.append(result['A'][np.random.randint(0,1000)])
        X_test.append(result['X'][np.random.randint(0,1000)])
        E_test.append(result['E'][np.random.randint(0,1000)])
    
    split_result = {
        'A_train' : A_train,
        'X_train' : X_train,
        'E_train' : E_train,
        'A_valid' : A_valid,
        'X_valid' : X_valid,
        'E_valid' : E_valid,
        'A_test' : A_test,
        'X_test' : X_test,
        'E_test' : E_test
    }

    return split_result
    
def convert_to_inputs_outputs_tf(AXEN_dict_subset, molecule_num, subset_num):

    A_list = []
    X_list = []
    output_E_list = []
    molecule_atom_num_list = []
    


    for i in range(molecule_num):
        A_hat = AXEN_dict_subset['molecule{}'.format(subset_num*1000 + i)][0]
        D_hat = np.diag(np.sum(A_hat, axis=0))
        D_inv = np.linalg.inv(D_hat)
        A = np.dot(D_inv, A_hat)

        A = tf.convert_to_tensor(A)
        # A is a tensor shape (29,29)

        N = AXEN_dict_subset['molecule{}'.format(subset_num*1000 + i)][3]
        N_array = [
            N
        ]
        N_array = np.asarray(N).reshape([-1,1])
        # N_array's shape is (1,1)

        # X normalisation
        X = AXEN_dict_subset['molecule{}'.format(subset_num*1000 + i)][1]
        X = X/maxX/N**2
        X = tf.convert_to_tensor(X)
        # X is a tensor shape (29,29)

        E = AXEN_dict_subset['molecule{}'.format(subset_num*1000 + i)][2]
        E = (-E-minE)/(maxE-minE)
        # E normalisation btw (0, 1)
        E = tf.convert_to_tensor(E)
        # E shape ()
        

        A_list.append(A)
        X_list.append(X)
        output_E_list.append(E)
        molecule_atom_num_list.append(N_array)

    # converting all to tensor
    A_list = tf.convert_to_tensor(A_list)
    X_list = tf.convert_to_tensor(X_list)
    output_E_list = tf.convert_to_tensor(output_E_list)
    result = {
        'A' : A_list,
        'X' : X_list,
        'E' : output_E_list,
        'N' : molecule_atom_num_list
    }


def convert_output(which_output, output_file_path, molecule_num_list):
    output_dict = {
        'G' : 'free_G',
        'Cv' : 'Cv',
        'H' : 'enthalpy'
    }
    output_file_path = output_file_path +'\\molecule_{}_dict.pickle'.format(output_dict[which_output])

    output_value_dict = load_dict(output_file_path)

    output_list = []

    for i in molecule_num_list:
        
        output = output_value_dict['molecule{}'.format(i)]
        output = (abs(output)-max_min_dict[which_output][1])/(max_min_dict[which_output][0]-max_min_dict[which_output][1])
        # output normalisation btw (0, 1)
        output_array = [
            output
        ]
        output_array = np.asarray(output_array).reshape([-1,1])
        
        output_list.append(output_array)

    return output_list