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

z_index = 0

#make Zebrafinch blocks into chunks of size 512x2048x2048 (smaller if on edge)
from functions import readData, makeFolder, blockFolderPath, writeData
zrange = np.arange(param.z_start,param.z_start+param.n_blocks_z)

sample_name = "labels_512_2048_2048"
folder_path = "/n/pfister_lab2/Lab/tfranzmeyer/Zebrafinch/"

for z_block in range(12):
    for y_block in range(3):
        filename = folder_path+"/"+sample_name+"/"+"/z"+str(z_block).zfill(2)+"y"+str(y_block).zfill(2)+"x"+str(0).zfill(2)
        block_0 = readData(box=[1],filename=filename)

        filename = folder_path+"/"+sample_name+"/"+"/z"+str(z_block).zfill(2)+"y"+str(y_block).zfill(2)+"x"+str(1).zfill(2)
        block_1 = readData(box=[1],filename=filename)

        filename = folder_path+"/"+sample_name+"/"+"/z"+str(z_block).zfill(2)+"y"+str(y_block).zfill(2)+"x"+str(2).zfill(2)
        block_2 = readData(box=[1],filename=filename)

        x_block = np.concatenate((block_0,block_1,block_2),axis=2)
	print(x_block.shape)

        if y_block == 0:
            y_block=x_block.copy()
        else:
            y_block = np.concatenate((y_block,x_block),axis=1)

        print(y_block.shape)

    for i in range(4):
        chunk = y_block[i*128:((i+1)*128),:,:]
        print(chunk.shape)
        writeData(folder_path+"/"+sample_name+"/"+str(z_index).zfill(4),chunk)
        z_index+=128
