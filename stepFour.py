import cc3d
import numpy as np
import time
import h5py
from numba import njit, types
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from numba.typed import Dict
import os
import pickle
import param
import sys

from functions import readData, writeData, compareOutp, concatBlocks

# (STEP 4 visualize wholes
# print out total of found wholes
blocks_concat = concatBlocks(z_start=param.z_start, y_start=param.y_start, x_start=param.x_start, n_blocks_z=param.n_blocks_z, n_blocks_y=param.n_blocks_y, n_blocks_x=param.n_blocks_x,
                                bs_z=param.max_bs_z, bs_y=param.max_bs_y, bs_x=param.max_bs_x, output_path=param.folder_path)

filename = param.data_path+"/"+param.sample_name+"/"+param.sample_name
box = [1]
labels_inp = readData(box, filename)
neg = np.subtract(blocks_concat, labels_inp)
output_name = "wholes"
writeData(param.folder_path+output_name, neg)

compareOutp(output_path=param.data_path,sample_name=param.sample_name,ID_B=param.outp_ID )
