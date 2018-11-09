import numpy as np
import pickle

def dict_to_A(molecule_dict, mols):
    A_dict = {}
    for n in range(mols):

        mol = molecule_dict['molecule%d' % n]
        connection_matrix_size = (29,29)
        A = np.zeros(connection_matrix_size)
    
        for i in range(29):
            for j in range(29):
                if i == j:
                    pass
                if i != j:
                    xi = mol[i, 1]
                    yi = mol[i, 2]
                    zi = mol[i, 3]
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
                    elif dist < 1.11:
                        # the longest C-H bonding length is ~1.10
                        # calculated value was 1.09
                        A[i,j] = 1
        A_dict['molecule%d' % n] = A
    return A_dict