import logging
import sqlite3
import pandas as pd
import bz2

"""get # of atoms for each molecule"""
logging.info("Accessing the database...")
# designating db file path
sqlite_file = 'C:\KT_project\dataset\gdb9.db'
# give connection to the db and get data of molecule
conn = sqlite3.connect(sqlite_file)
c=conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = c.fetchall()
logging.info("tables are fetched.")
# systems_table 
logging.info("systems table are extracted...")
molecule_table = pd.read_sql_query("SELECT * from %s" % 'systems', conn)
molecule_table = pd.DataFrame(molecule_table, columns = [
    'id',
    'unique_id',
    'numbers',
    'cell',
    'pbc',
    'key_value_pairs',
    'energy',
    'natoms'
])
logging.info("finished extracting system table")

natoms = molecule_table['natoms']
totalnmolecule = 133885
totalnatoms = 0
for i in range(totalnmolecule):
    totalnatoms += natoms[i]
# natoms, totalnmolecule 얻어짐
# totalnatoms = 2407753

""" extracting data from bz2 file """

myfile = 'C:\KT_project\dataset\dsgdb9nsd.xyz.tar.bz2'
txt_file = 'C:/KT_project/dataset/file.txt'
excel_file = 'C:\KT_project\dataset\data_in_excel.xlsx'

def extract_data_from_bz2(rawdb, txt_to_be_written, nmolecule, totalnatom):
    stream = bz2.BZ2File(rawdb)
    with open(txt_to_be_written, 'w') as f:
        for i in range(5*nmolecule+totalnatom):
            f.write(str(stream.readline())+"\n")

bz2_to_txt = input("Do you need to convert bz2 file to txt file? [y / N]")
if bz2_to_txt == 'y':
    extract_data_from_bz2(myfile, txt_file, totalnmolecule, totalnatoms)
else:
    pass

""" extracting only the atom type, position(x,y,z), partial charges from the text data """

# natoms = natoms, totalnmolecule = totalnmolecule
def db_frame_construction(natoms, totalnatoms, totalnmolecule, txt_file_path, excel_file_path):
    mol_num = []
    atom_num = []
    for i in range(totalnmolecule):
        for n in range(natoms[i]):
            mol_num.append(i)
            atom_num.append(n)
    print("successfully made molecule index column & atom index column!")
    # achieved molecule number column & atom number column
    
    f = open(txt_file_path, 'r')
    lines = f.readlines()
    atom_info = []
    count_num=2
    total_num_lines = 5 * totalnmolecule + totalnatoms
    for i in range(totalnmolecule):
        for k in range(natoms[i]):
            atom_info.append(lines[count_num+k])
        count_num = count_num + natoms[i] + 5
    
    f.close()
    print("successfully put all the data in a single list")
    # achieved atom info for each atom in list
    # however, the item is presented like this
    # "b'C\\t-0.0126981359\\t 1.0858041578\\t 0.0080009958\\t-0.535689\\n'\n"

    atom_type_info = []
    atom_x_info = []
    atom_y_info = []
    atom_z_info = []
    atom_mu_info = []

    for i in range(totalnatoms):
        sliced_info = atom_info[i].split('\\')
        atom_type_info.append(sliced_info[0][-1])
        atom_x_info.append(sliced_info[1][1:])
        atom_y_info.append(sliced_info[2][1:])
        atom_z_info.append(sliced_info[3][1:])
        atom_mu_info.append(sliced_info[4][1:])
# TO-DO : should convert string x,y,z into float

    db_data = {
        'mol_num' : mol_num,
        'atom_num' : atom_num,
        'atom_type_info' : atom_type_info,
        'atom_x_info' : atom_x_info,
        'atom_y_info' : atom_y_info,
        'atom_z_info' : atom_z_info,
        'atom_mu_info' : atom_mu_info,
        }

    print("Getting the data into data frame...")
    db_data_frame = pd.DataFrame(db_data, columns=[
        'mol_num',
        'atom_num',
        'atom_type_info',
        'atom_x_info',
        'atom_y_info',
        'atom_z_info',
        'atom_mu_info'])
    print("all molecule data are now in the data frame.")
    
    """
    should write the date by deviding the data into subset
        
        num = 0
        for i in range(number of the molecule you want to put in one subset):
            num += natoms[i]
        
        subset_db = db_data_frame[0:num]
        subset_db.to_excel(excel_file_path)
    remind that the num value is different for every iteration
    
    print("writing the data into excel file.")
    db_data_frame.to_excel(excel_file_path)
    print("all molecule data has been wriiten in excel file")
    """

txt_to_excel = input("Do you need to convert txt file to excel file? [y / N]")
if txt_to_excel == 'y':
    atom_info_df = db_frame_construction(natoms,totalnatoms, totalnmolecule, txt_file, excel_file)
else:
    pass


