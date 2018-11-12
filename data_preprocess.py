import logging
import sqlite3
import pandas as pd

logging.info("Accessing the database...")
# designating db file path
sqlite_file = 'C:\KT_project\dataset\gdb9.db'
# give connection
conn = sqlite3.connect(sqlite_file)
c=conn.cursor()

# check the structure of db
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = c.fetchall()
logging.info("tables are now fetched.")

"""checked the table name as below"""
# for table_name in tables:
#     table_name=table_name[0]
#     print(table_name)
"""
table names are as follows
systems
sqlite_sequence
species
keys
text_key_values
number_key_values
information
"""

logging.info("systems table are extracted...")
# systems_table 
systems_table = pd.read_sql_query("SELECT * from %s" % 'systems', conn)

"""
columns in systems table
acquired by systems_table.columns
['id', 'unique_id', 'ctime', 'mtime', 'username', 'numbers', 'positions',
       'cell', 'pbc', 'initial_magmoms', 'initial_charges', 'masses', 'tags',
       'momenta', 'constraints', 'calculator', 'calculator_parameters',
       'energy', 'free_energy', 'forces', 'stress', 'dipole', 'magmoms',
       'magmom', 'charges', 'key_value_pairs', 'data', 'natoms', 'fmax',
       'smax', 'volume', 'mass', 'charge']
"""

systems_table = pd.DataFrame(systems_table, columns = [
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
"""
key-value for each atom
["rcC", "r2", "gap", "rcA", "mu", "rcB", "homo"
, "energy_U0", "Cv", "enthalpy_H", "free_G", "alpha", "energy_U"
, "lumo", "zpve"]
"""