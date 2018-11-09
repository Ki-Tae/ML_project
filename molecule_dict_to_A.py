import numpy as np
import pickle

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
    '7#8' : 1.06
} 
offset = 0.01

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
        else:
            pass
    
    return connected

def dict_to_A(molecule_dict, mols):
    A_dict = {}
    X_dict = {}
    for n in range(mols):

        mol = molecule_dict['molecule%d' % n]
        connection_matrix_size = (29,29)
        A = np.zeros(connection_matrix_size)
        X = np.zeros(connection_matrix_size)

        for i in range(29):
            for j in range(29):
                if i == j:
                    pass
                if i != j:
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
                    
                    if (xi, yi, zi) == (0, 0, 0):
                        pass
                    elif (xj, yj, zj) == (0, 0, 0):
                        pass
                    else:
                        connected = connection_detector(dist=dist, typei=typei, typej=typej,bondlength=bl_table, offset=offset)
                        A[i,j] = connected
                    
                    
        A_dict['molecule%d' % n] = A
    return A_dict


