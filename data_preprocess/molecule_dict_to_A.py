import numpy as np
import pickle
import csv

bl_table = {
    # single bond
    '1-6' : 1.09,
    '1-7' : 1.01,
    '1-8' : 0.96,
    '1-9' : 1.27,
    '6-6' : 1.54,
    '6-7' : 1.47,
    '6-8' : 1.43,
    '6-9' : 1.33,
    '7-7' : 1.46,
    '7-8' : 1.44,
    '7-9' : 1.39,
    '8-8' : 1.48,
    '8-9' : 1.42,
    '9-9' : 1.43,
    # double bond
    '6=6' : 1.34,
    '6=7' : 1.27,
    '6=8' : 1.23,
    '7=7' : 1.22,
    '7=8' : 1.20,
    # triple bond
    '6#6' : 1.21,
    '6#7' : 1.15,
    '6#8' : 1.13,
    '7#7' : 1.10,
    '7#8' : 1.06,
    # ring bond
    '6~6' : 1.38,
    '6~7' : 1.33,
    '7~7' : 1.37
    
} 
offset = 0.12

def connection_detector(dist, typei, typej, bondlength, offset):
    atom_type = [typei, typej]
    atom_type = sorted(atom_type)
    connected = 0
    # not connected at first
    if '%d-%d' % (atom_type[0], atom_type[1]) in bondlength:
    # check if bonding is valid
        if bondlength['%d-%d' % (atom_type[0], atom_type[1])] - offset <= dist <= bondlength['%d-%d' % (atom_type[0], atom_type[1])] + offset:
            connected = 1
            # if sigle-bonded
        elif '%d=%d' % (atom_type[0], atom_type[1]) in bondlength:
            if bondlength['%d=%d' % (atom_type[0], atom_type[1])] - offset <= dist <= bondlength['%d=%d' % (atom_type[0], atom_type[1])] + offset:
                connected = 1
            # if double-bonded
            else:
                pass
        elif '%d#%d' % (atom_type[0], atom_type[1]) in bondlength:
            if bondlength['%d=%d' % (atom_type[0], atom_type[1])] - offset <= dist <= bondlength['%d=%d' % (atom_type[0], atom_type[1])] + offset:
                connected = 1
            # if triple-bonded
            else:
                pass
        elif '%d~%d' % (atom_type[0], atom_type[1]) in bondlength:
            if bondlength['%d~%d' % (atom_type[0], atom_type[1])] - offset <= dist <= bondlength['%d~%d' % (atom_type[0], atom_type[1])] + offset:
                connected = 1
            # if resonance
            else:
                pass
        else:
            pass
    
    return connected

def dict_to_AXEN(molecule_dict, outputs, mol_num_dict, nthsubset):
    AXEN_dict = {}
    
    for n in range(nthsubset*1000, (nthsubset+1)*1000):

        mol = molecule_dict['molecule%d' % n]
        connection_matrix_size = (29,29)
        A = np.zeros(connection_matrix_size)
        X = np.zeros(connection_matrix_size)

        for i in range(29):
            for j in range(29):
                if i == j:
                    # for constructing A
                    A[i, j] = 1
                    # for constructing X
                    zi = mol[i, 0]
                    X[i, j] = 0.5*(zi**(2.4))

                elif i != j:
                    
                    typei = mol[i, 0]
                    xi = mol[i, 1]
                    yi = mol[i, 2]
                    zi = mol[i, 3]

                    typej = mol[j, 0]
                    xj = mol[j, 1]
                    yj = mol[j, 2]
                    zj = mol[j, 3]

                    positioni = np.array((xi, yi, zi))
                    positionj = np.array((xj, yj, zj))
                    dist = np.linalg.norm(positioni-positionj)
                    
                    # for constructing A
                    if (xi, yi, zi) == (0, 0, 0):
                        pass
                    elif (xj, yj, zj) == (0, 0, 0):
                        pass
                    else:
                        connected = connection_detector(dist=dist, typei=typei, typej=typej,bondlength=bl_table, offset=offset)
                        A[i,j] = connected
                    
                    # for constructing X
                    if (xi, yi, zi) == (0, 0, 0):
                        pass
                    elif (xj, yj, zj) == (0, 0, 0):
                        pass
                    else:
                        X[i, j] = (typei*typej)/dist
        E = outputs['molecule%d' % n]
        N = mol_num_dict['molecule%d' % n]
        AXEN_dict['molecule%d' % n] = [A, X, E, N]                                
    return AXEN_dict

def dict_to_muX(molecule_dict, nthsubset):
    muX_dict = {}
    for n in range(nthsubset*1000, (nthsubset+1)*1000):

        mol = molecule_dict['molecule%d' % n]
        connection_matrix_size = (29,29)
        muX = np.zeros(connection_matrix_size)

        for i in range(29):
            for j in range(29):
                if i == j:
                    # for constructing muX
                    mui = mol[i, 4]
                    muX[i, j] = 0.5*(mui**(2.4))

                elif i != j:
                    
                    xi = mol[i, 1]
                    yi = mol[i, 2]
                    zi = mol[i, 3]
                    mui = mol[i, 4]
                    
                    xj = mol[j, 1]
                    yj = mol[j, 2]
                    zj = mol[j, 3]
                    muj = mol[j, 4]

                    positioni = np.array((xi, yi, zi))
                    positionj = np.array((xj, yj, zj))
                    dist = np.linalg.norm(positioni-positionj)

                    # for constructing muX
                    if (xi, yi, zi) == (0, 0, 0):
                        pass
                    elif (xj, yj, zj) == (0, 0, 0):
                        pass
                    else:
                        muX[i, j] = (mui*muj)/dist                    
                    
        muX_dict['molecule%d' % n] = muX
    return muX_dict

def saving_dict_to_csv(A_dict, X_dict, nthsubset):
    # writing the A_dict into csv file
    with open('C:\KT_project\dataset\A_subsets\A_dict_subset%d.csv' % nthsubset,'w') as f:
        w = csv.writer(f)
        w.writerow(A_dict.keys())
        w.writerow(A_dict.values())
    
    # writing the X_dict into csv file 
    with open('C:\KT_project\dataset\X_subsets\X_dict_subset%d.csv' % nthsubset,'w') as f:
        w = csv.writer(f)
        w.writerow(X_dict.keys())
        w.writerow(X_dict.values())
    """
    # writing the X_dict into csv file 
    with open('C:\KT_project\dataset\muX_subsets\muX_dict_subset%d.csv' % nthsubset,'w') as f:
        w = csv.writer(f)
        w.writerow(muX_dict.keys())
        w.writerow(muX_dict.values())
    """    

def saving_dict_to_pickle(AXEN_dict, nthsubset):
    with open('C:\KT_project\dataset\AXEN_dict_subset\AXEN_dict_subset%d.pickle' % nthsubset,'wb') as handle:
        pickle.dump(AXEN_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
