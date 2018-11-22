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

def convert_to_inputs_outputs(AXE_dict_subset, molecule_num):
    input_x = []
    output_x = []
    for i in range(molecule_num):
        input_x.append(AXE_dict_subset['molecule{}'.format(i)][1])
        output_x.append(AXE_dict_subset['molecule{}'.format(i)][2])
        
    # input_x = np.asarray(input_x)
    # input shape (molecule_num, 29, 29)
    input_x = np.swapaxes(input_x, 0, 2)
    # input shape (29,29, molecule_num)

    output_x = np.asarray(output_x)
    # output shape (molecule_num, ) is gained
    return input_x, output_x





