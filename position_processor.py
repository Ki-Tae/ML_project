import pandas as pd
import numpy as np

# coverting x,y,z position to distance vector (adjacency vector)

#number of molecule in data subset
num = 30
converting_table = {
    'C' : 6,
    'H' : 1,
    'O' : 8,
    'N' : 7,
    'F' : 9
}
excel_file = 'C:\KT_project\dataset\data_subset.xlsx'

# def convert_xyz_to_distance_in_nth_molecule(excel_file):
#     try:
#         data = pd.read_excel(excel_file)
#     except:
#         NameError
#         print("Excel file is not loaded.")

# n = number of molecule defined above
def molecule_info_matrix(excel_file, converting_table, n):
    data = pd.read_excel(excel_file)
    data = data.loc[data['mol_num'] == n]
    #n-th molecule data is retrieved to data
    row_num = data.shape[0]

    matrix_size = (29,5)
    molecule_data = np.zeros(matrix_size)
    for i in range(row_num):
        atom_data = data.loc[data['atom_num'] == i]
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

molecules_dict = {}

for i in range(num):
    matrix = molecule_info_matrix(excel_file=excel_file, converting_table=converting_table, n=i)
    molecules_dict['molecule%d' % i ] = matrix
    # molecules dictionary for the subset data is achieved

# writing the file into csv
"""
with open('C:\KT_project\dataset/molecule_dict.csv', 'w') as f:
    for i in range(30):
        f.write("%s, %d, %d, %d, %d, %d\n" %
        ('molecule%d' % i,
        molecules_dict['molecule%d' % i][0],
        molecules_dict['molecule%d' % i][1],
        molecules_dict['molecule%d' % i][2],
        molecules_dict['molecule%d' % i][3],
        molecules_dict['molecule%d' % i][4]))
"""    

    

