import cc3d
import numpy as np
import time
import h5py
from numba import njit, types
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from numba.typed import Dict
import os
import sys
import pickle
import param

# bs_z = 512
bs_y = 2048
bs_x = 2048


#make Zebrafinch blocks into chunks of size 512x2048x2048 (smaller if on edge)
from functions import readData, makeFolder, blockFolderPath, writeData
zrange = np.arange(param.z_start,param.z_start+param.n_blocks_z)

for bz in zrange[::4]:

    print( "bz is: " + str(bz), flush=True)
    z_block = 0

    filename = param.folder_path+"/"+param.sample_name+"/"+str(bz*128).zfill(4)
    block_a = readData(box=[1],filename=filename)

    if bz != zrange[-1]:
        filename = param.folder_path+"/"+param.sample_name+"/"+str((bz+1)*128).zfill(4)
        block_b = readData(box=[1],filename=filename)
        block_a = np.concatenate((block_a,block_b), axis=0)
        del block_b

        filename = param.folder_path+"/"+param.sample_name+"/"+str((bz+2)*128).zfill(4)
        block_c = readData(box=[1],filename=filename)
        block_a = np.concatenate((block_a,block_c), axis=0)
        del block_c

        filename = param.folder_path+"/"+param.sample_name+"/"+str((bz+3)*128).zfill(4)
        block_d = readData(box=[1],filename=filename)
        block_a = np.concatenate((block_a,block_d), axis=0)
        del block_d

    for y_block in range(3):
        for x_block in range(3):
            # outp_folder = blockFolderPath(param.folder_path,z_block,y_block,x_block)
            chunk = block_a[:,y_block:(y_block+1):bs_y,x_block:(x_block+1):bs_x]
            print(y_block,x_block)
            print(chunk.shape)
            writeData(param.folder_path+"/z"+str(z_block).zfill(2)+"y"+str(y_block).zfill(2)+"x"+str(x_block).zfill(2),chunk)
