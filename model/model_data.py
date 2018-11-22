import numpy as np
import pickle

# loading the data
"""
define how many molecules to put for 1 batch
molecule_num = 10
molecule_num = 10
AXE_file_path = "C:\KT_project\dataset\AXE_dict_subset\AXE_dict_subset0.pickle"
"""



def load_dict(AXE_file_path):
    with open(AXE_file_path, 'rb') as f:
        AXE_dict_subset = pickle.load(f)

    return AXE_dict_subset

def convert_to_inputs_outputs(AXE_dict_subset, molecule_num, subset_num):
    input_array_list = []
    output_E_list = []

    for i in range(molecule_num):
        input_array = [
            AXE_dict_subset['molecule{}'.format(subset_num*1000 + i)][0],
            AXE_dict_subset['molecule{}'.format(subset_num*1000 + i)][1]
        ]
        input_array = np.asarray(input_array)
        # input_array's shape is (2,29,29)

        input_array_list.append(input_array)
        output_E_list.append(AXE_dict_subset['molecule{}'.format(subset_num*1000 + i)][2])
        
    return input_array_list, output_E_list





