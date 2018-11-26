import numpy as np
import pickle

# loading the data
"""
each X are normalized differently.
"""

"""
max U0 ['molecule0'] = -1101.4877900833399
min U0 ['molecule133620'] = -19444.3873485546
"""
minE = 1101.4877900833399
maxE = 19444.3873485546
maxX = 0.5*9**2.4

min_muX = -0.5
max_muX = 0.5

def load_dict(file_path):
    with open(file_path, 'rb') as f:
        dict_subset = pickle.load(f)

    return dict_subset

def convert_to_inputs_outputs(AXEN_dict_subset, molecule_num, subset_num):
    A_array_list = []
    X_array_list = []
    output_E_list = []
    molecule_atom_num_list = []
    


    for i in range(molecule_num):
        A_hat = AXEN_dict_subset['molecule{}'.format(subset_num*1000 + i)][0]
        D_hat = np.diag(np.sum(A_hat, axis=0))
        D_inv = np.linalg.inv(D_hat)
        A = np.dot(D_inv, A_hat)

        A_array = [
            A
        ]
        A_array = np.asarray(A_array)
        # A_array's shape is (1,29,29)

        N = AXEN_dict_subset['molecule{}'.format(subset_num*1000 + i)][3]
        N_array = [
            N
        ]
        N_array = np.asarray(N).reshape([-1,1])
        # N_array's shape is (1,1)

        # X normalisation
        X = AXEN_dict_subset['molecule{}'.format(subset_num*1000 + i)][1]
        X = X/maxX/N**2
        X_array = [
            X
        ]
        X_array = np.asarray(X_array)
        # X_array's shape is (1,29,29)

        E = AXEN_dict_subset['molecule{}'.format(subset_num*1000 + i)][2]
        E = (-E-minE)/(maxE-minE)
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

def convert_to_muX(AXEN_dict_subset, muX_dict_subset, molecule_num, subset_num):
    muX_array_list = []
    
    for i in range(molecule_num):
        
        N = AXEN_dict_subset['molecule{}'.format(subset_num*1000 + i)][3]
        
        # muX normalisation
        muX = muX_dict_subset['molecule{}'.format(subset_num*1000 + i)]
        muX = muX/N**2
        # max_muX - min_muX = 1
        muX_array = [
            muX
        ]
        muX_array = np.asarray(muX_array, object)
        # muX_array's shape is (1,29,29)

        muX_array_list.append(muX_array)
    
    return  muX_array_list

def convert_to_invarient_X(AXEN_dict_subset, molecule_num, subset_num):
    inv_X_array_list = []

    for i in range(molecule_num):
        # making X invarient
        
        X = AXEN_dict_subset['molecule{}'.format(subset_num*1000 + i)][1]
        
        X = X/maxX/N**2
        X_array = [
            X
        ]
        X_array = np.asarray(X_array)
        # X_array's shape is (1,29,29)