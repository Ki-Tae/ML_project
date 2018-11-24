import numpy as np
import pickle

# loading the data
"""
define how many molecules to put for 1 batch
molecule_num = 10
molecule_num = 10
AXE_file_path = "C:\KT_project\dataset\AXE_dict_subset\AXE_dict_subset0.pickle"
"""
max_X = 9**2.4/2*841
Ha_to_ev = 1000

def load_dict(AXE_file_path):
    with open(AXE_file_path, 'rb') as f:
        AXE_dict_subset = pickle.load(f)

    return AXE_dict_subset

def convert_to_inputs_outputs(AXE_dict_subset, molecule_num, subset_num):
    A_array_list = []
    X_array_list = []
    output_E_list = []

    for i in range(molecule_num):
        A_hat = AXE_dict_subset['molecule{}'.format(subset_num*1000 + i)][0]
        D_hat = np.diag(np.sum(A_hat, axis=0))
        D_inv = np.linalg.inv(D_hat)
        A = np.dot(D_inv, A_hat)

        A_array = [
            A
        ]
        A_array = np.asarray(A_array)
        # A_array's shape is (1,29,29)

        X = AXE_dict_subset['molecule{}'.format(subset_num*1000 + i)][1]/max_X
        X_array = [
            X
        ]
        X_array = np.asarray(X_array)
        # X_array's shape is (1,29,29)

        E = AXE_dict_subset['molecule{}'.format(subset_num*1000 + i)][2]/Ha_to_ev
        E_array = [
            E
        ]
        E_array = np.asarray(E_array).reshape([-1,1])
        # E_array's shape is (1,1)

        A_array_list.append(A_array)
        X_array_list.append(X_array)
        output_E_list.append(E_array)
        
    result = {
        'A' : A_array_list,
        'X' : X_array_list,
        'E' : output_E_list
    }

    return  result