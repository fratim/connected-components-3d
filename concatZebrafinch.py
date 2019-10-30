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

sample_name = "stacked_512_2048_2048"
folder_path = "/n/pfister_lab2/Lab/tfranzmeyer/Zebrafinch/"

for z_block in range(12):
    print("Z is " + str(z_block))

    for y_block in range(3):
        print("Y is " + str(y_block))

        filename = folder_path+"/"+sample_name+"/"+"/z"+str(z_block).zfill(2)+"y"+str(y_block).zfill(2)+"x"+str(0).zfill(2)
        block_0 = readData(box=[1],filename=filename)
        print(block_0.shape)

        filename = folder_path+"/"+sample_name+"/"+"/z"+str(z_block).zfill(2)+"y"+str(y_block).zfill(2)+"x"+str(1).zfill(2)
        block_1 = readData(box=[1],filename=filename)
        print(block_1.shape)

        filename = folder_path+"/"+sample_name+"/"+"/z"+str(z_block).zfill(2)+"y"+str(y_block).zfill(2)+"x"+str(2).zfill(2)
        block_2 = readData(box=[1],filename=filename)
        print(block_2.shape)

        x_data = np.concatenate((block_0,block_1,block_2),axis=2)
        print("X Block:")
        print(x_data.shape)

        if y_block == 0:
            y_data=x_data.copy()
        else:
            y_data = np.concatenate((y_data,x_data),axis=1)

        print("Y Block:")
        print(y_data.shape)

    for i in range(4):
        chunk = y_data[i*128:((i+1)*128),:,:]
        print("Chunk " + str(i))
        print(chunk.shape)
        writeData(folder_path+"/"+sample_name+"/"+str(z_index).zfill(4),chunk)
        z_index+=128

    del chunk, block_0, block_1, block_2, x_data, y_data
