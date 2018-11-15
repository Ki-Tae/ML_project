import pickle

# loading the data

def load_dict(A_dict_file_path, X_dict_file_path):
    with open(A_dict_file_path, 'rb') as f:
        A_dict_subset = pickle.load(f)

    with open(X_dict_file_path, 'rb') as f:
        X_dict_subset = pickle.load(f)

    return A_dict_subset, X_dict_subset




