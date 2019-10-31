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
zrange = np.arange(0,45)

sample_name = ""
folder_path = "/n/pfister_lab2/Lab/tfranzmeyer/Zebrafinch/"

for bz in zrange[::4]:

    z_block = int(bz/4)
    print( "bz is: " + str(bz), flush=True)

    filename = folder_path+"/"+sample_name+"/"+str(bz*128).zfill(4)
    block_a = readData(box=[1],filename=filename)

    if bz != zrange[-1]:
        filename = folder_path+"/"+sample_name+"/"+str((bz+1)*128).zfill(4)
        block_b = readData(box=[1],filename=filename)
        block_a = np.concatenate((block_a,block_b), axis=0)
        del block_b

        filename = folder_path+"/"+sample_name+"/"+str((bz+2)*128).zfill(4)
        block_c = readData(box=[1],filename=filename)
        block_a = np.concatenate((block_a,block_c), axis=0)
        del block_c

        filename = folder_path+"/"+sample_name+"/"+str((bz+3)*128).zfill(4)
        block_d = readData(box=[1],filename=filename)
        block_a = np.concatenate((block_a,block_d), axis=0)
        del block_d

    for y_block in range(3):
        for x_block in range(3):
            # outp_folder = blockFolderPath(folder_path,z_block,y_block,x_block)
            chunk = block_a[:,y_block*bs_y:(y_block+1)*bs_y,x_block*bs_x:(x_block+1)*bs_x]
            print(y_block,x_block)
            print(chunk.shape)
            writeData(folder_path+"/z"+str(z_block).zfill(2)+"y"+str(y_block).zfill(2)+"x"+str(x_block).zfill(2),chunk)
