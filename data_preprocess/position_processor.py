import pandas as pd
import numpy as np
from molecule_dict_to_A import dict_to_A, dict_to_muX, saving_dict_to_csv

# coverting x,y,z position to distance vector (adjacency vector)

# number of molecule in data subset / only 133rd subset has 885 molecules
num = 1000
converting_table = {
    'C' : 6,
    'H' : 1,
    'O' : 8,
    'N' : 7,
    'F' : 9
}


# n = number of molecule defined above
def molecule_info_matrix(excel_data, converting_table, n):
    nth_molecule_data = excel_data.loc[excel_data['mol_num'] == n]
    #n-th molecule data is retrieved to data
    row_num = nth_molecule_data.shape[0]

    matrix_size = (29,5)
    molecule_data = np.zeros(matrix_size)
    for i in range(row_num):
        atom_data = nth_molecule_data.loc[nth_molecule_data['atom_num'] == i]
        if atom_data.get('atom_type_info').item() == 'C':
            molecule_data[i,0] = converting_table['C']
        elif atom_data.get('atom_type_info').item() == 'H':
            molecule_data[i,0] = converting_table['H']
        elif atom_data.get('atom_type_info').item() == 'O':
            molecule_data[i,0] = converting_table['O']
        elif atom_data.get('atom_type_info').item() == 'N':
            molecule_data[i,0] = converting_table['N']
        elif atom_data.get('atom_type_info').item() == 'F':
            molecule_data[i,0] = converting_table['F']
        molecule_data[i,1] = atom_data.get('atom_x_info').item()
        molecule_data[i,2] = atom_data.get('atom_y_info').item()
        molecule_data[i,3] = atom_data.get('atom_z_info').item()
        molecule_data[i,4] = atom_data.get('atom_mu_info').item()
    
    return molecule_data

"""
molecules_dict = {}

for i in range(num):
    matrix = molecule_info_matrix(excel_data = data, converting_table=converting_table, n=i)
    molecules_dict['molecule%d' % i ] = matrix
    # molecules dictionary for the subset data is achieved

A_dict, X_dict = dict_to_A(molecule_dict = molecules_dict, mols = num)
saving_dict_to_csv(A_dict=A_dict, X_dict=X_dict, nthsubset = 0)
"""

# excel_file = 'C:\KT_project\dataset\\raw_subset\subset0.xlsx'
# data = pd.read_excel(excel_file)

def altering_nth_subset_to_A_X(excel_data, nthsubset):
    molecules_dict={}
    for i in range(num):
        matrix = molecule_info_matrix(excel_data = excel_data, converting_table = converting_table, n=i)
        molecules_dict['molecule%d' % (nthsubset*1000 + i)] = matrix
        # molecules dictionary for the subset data is achieved
    print("molecules dictionary for subset%d is obtained" % nthsubset)

    # muX_dict = dict_to_muX(molecule_dict = molecules_dict, nthsubset = nthsubset)    
    A_dict, X_dict = dict_to_A(molecule_dict = molecules_dict, nthsubset = nthsubset)
    saving_dict_to_csv(A_dict=A_dict, X_dict=X_dict, nthsubset = nthsubset)

for subsetnum in range(1,2):
    excel_file = 'C:\KT_project\dataset\\raw_subset\subset%d.xlsx' % subsetnum
    data = pd.read_excel(excel_file)
    altering_nth_subset_to_A_X(excel_data = data, nthsubset = subsetnum)
    print("subset%d now converted to A, X data." % subsetnum)


