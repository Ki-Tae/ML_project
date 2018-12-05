from model_data import load_dict, convert_to_inputs_outputs, convert_to_muX, convert_output
import numpy as np
import pickle
"""
custom iterator for gcn model
1. load dict randomly from the entire dataset
2. draw A,X,E,N with desired option (ex)num of molecule
3. give option to change the output.
3. process the info into feedable form
"""
"""
4. train & validation test > record the data
5. test
"""
output_list = ['U', 'G', 'Cv', 'H']
output_file_path = 'C:\KT_project\\dataset\\output_values'

AXEN_dict_path = "C:\KT_project\dataset\AXEN_dict_subset"
other_output_path = "C:\KT_project\dataset\output_values"
muX_dict_path = "C:\KT_project\dataset\muX_subsets"

def custom_iterator(which_input, which_output, atom_num_range):
    if which_input == 'AXEN':
        # here select random subset
        ran_subset_num = np.random.randint(0,133)
        subset_path = AXEN_dict_path + '\\AXEN_dict_subset{}.pickle'.format(ran_subset_num)
        subset = load_dict(subset_path)
        # subset is achieved
        min_num = min(atom_num_range)
        max_num = max(atom_num_range) + 1
        # extracting molecules in desired range
        filtered_subset = {}
        molecule_list = []
        for i in range(1000):
            if min_num <= subset['molecule{}'.format(ran_subset_num*1000 + i)][3] <= max_num:
                filtered_subset['molecule{}'.format(ran_subset_num*1000 + i)] = subset['molecule{}'.format(ran_subset_num*1000 + i)]
                molecule_list.append(ran_subset_num*1000 + i)
        length_num = len(molecule_list)
        
        AXEN = convert_to_inputs_outputs(AXEN_dict_subset = filtered_subset, molecule_num_list = molecule_list, subset_num = ran_subset_num)
        
        result = {
            'AXEN' : AXEN,
            'length' : length_num
        }

        if which_output != 'U':
            output = convert_output(which_output=which_output, output_file_path = output_file_path, molecule_num_list=molecule_list)
            result['output'] = output

        return result
    # molecule list of desired length is obtained
    
    elif which_input == "muX":
        ran_num = np.random.randint(0,134)
        subset_path = AXEN_dict_path + '\\AXEN_dict_subset{}.pickle'.format(ran_num)
        subset = load_dict(subset_path)
        muX_subset_path = muX_dict_path + '\\muX_dict_subset{}.pickle'.format(ran_num)
        muX_subset = load_dict(muX_subset_path)
        # subset is achieved
        min_num = min(atom_num_range)
        max_num = max(atom_num_range) + 1

        filtered_subset = {}
        muX_filtered_subset = {}
        molecule_list = []
        
        for i in range(1000):
            if min_num <= subset['molecule{}'.format(ran_num*1000 + i)][3] <= max_num:
                filtered_subset['molecule{}'.format(ran_num*1000 + i)] = subset['molecule{}'.format(ran_num*1000 + i)]
                muX_filtered_subset['molecule{}'.format(ran_num*1000 + i)] = muX_subset['molecule{}'.format(ran_num*1000 + i)]
                molecule_list.append(ran_num*1000 + i)
        
        result = convert_to_inputs_outputs(AXEN_dict_subset = filtered_subset, molecule_num_list = molecule_list, subset_num = ran_num)
        mux_result = convert_to_muX(AXEN_dict_subset=filtered_subset, muX_dict_subset = muX_filtered_subset, molecule_num_list = molecule_list, subset_num = ran_num)
        length_num = len(molecule_list)

        return result, mux_result, length_num