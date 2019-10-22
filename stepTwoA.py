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
import math


from functions import readFromFile, findAdjLabelSetGlobal, dumpNumbaDictToFile, makeFolder, dumpToFile, IdiToIdx

# pass arguments
if(len(sys.argv))!=2:
    raise ValueError(" Scripts needs exactley 1 input arguments (bz by bx)")
else:
    bz_global = int(sys.argv[1])

# Iteration 1 interconnects 2 z-blocks, also interconnecting in all x-y directions and loading the needed variables
iterations_needed = param.iterations_needed
iteration = 1
z_range = np.arange(param.z_start, param.z_start+param.n_blocks_z)

print("------------------------------------------------", flush=True)
print("Iteration " + str(iteration), flush=True)
print("bz_global is: " + str(bz_global), flush=True)

if bz_global not in z_range[::2]:
    print("Block not computed in this iteration, aborted. ", flush=True)

elif bz_global in z_range[::2]:
    print(str(bz_global) + " is in z_range")

    border_comp_combined = Dict.empty(key_type=types.int64,value_type=types.int64)
    border_comp_exist_combined = set()
    neighbor_label_set_border_global = {(1,1)}

    # determine if the last block is single (uneven number of blocks)
    if bz_global+1>z_range[-1]:
        bz_range = [bz_global]
        isSingle = True
    elif bz_global+1<=z_range[-1]:
        bz_range = [bz_global, bz_global+1]
        isSingle = False
    else:
        raise ValueError("Unknown Error!")

    for bz in bz_range:
        for by in range(param.y_start, param.y_start+param.n_blocks_y):
            for bx in range(param.x_start, param.x_start+param.n_blocks_x):

                output_folder = param.folder_path+"/z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)+"/"

                border_comp_local = readFromFile("border_comp_local", output_folder, "")
                border_comp_exist_local = readFromFile("border_comp_exist_local", output_folder, "")

                border_comp_combined.update(border_comp_local)
                border_comp_exist_combined = border_comp_exist_combined.union(border_comp_exist_local)

                del border_comp_local, border_comp_exist_local

    border_comp_combined_new = Dict.empty(key_type=types.int64,value_type=types.int64)
    border_comp_exist_combined_new = {(2**30)}

    print("bz_range: " + str(bz_range), flush=True)
    for bz in bz_range:

        # choose z directions to connect
        if isSingle:
            connectInPosZdirec = False
            connectInNegZdirec = False

        # if first of two, connect in positive but not in negative direction
        elif bz == bz_range[0]:
            connectInPosZdirec = True
            connectInNegZdirec = False
        elif bz == bz_range[1]:
            connectInPosZdirec = False
            connectInNegZdirec = True

        else:
            raise ValueError("Unkown Error")

        for by in range(param.y_start, param.y_start+param.n_blocks_y):
            for bx in range(param.x_start, param.x_start+param.n_blocks_x):

                # find box to iterative over all blocks
                box = [bz*param.bs_z,(bz+1)*param.bs_z,by*param.bs_y,(by+1)*param.bs_y,bx*param.bs_x,(bx+1)*param.bs_x]
                border_comp_combined_new, border_comp_exist_combined_new, neighbor_label_set_border_global = findAdjLabelSetGlobal(box, neighbor_label_set_border_global,
                                                        border_comp_combined, border_comp_exist_combined, param.yres, param.xres, connectInPosZdirec, connectInNegZdirec,
                                                        border_comp_combined_new, border_comp_exist_combined_new)

    border_comp_exist_combined_new.remove((2**30))
    neighbor_label_set_border_global.remove((1,1))

    #compute coutput blocksize
    if isSingle:
        box_combined = [bz_global*param.bs_z,(bz_global+1)*param.bs_z,param.y_start*param.bs_y,(param.y_start+param.n_blocks_y)*param.bs_y,
                            param.x_start*param.bs_x,(param.x_start+param.n_blocks_x)*param.bs_x]
    elif not isSingle:
        box_combined = [bz_global*param.bs_z,(bz_global+2)*param.bs_z,param.y_start*param.bs_y,(param.y_start+param.n_blocks_y)*param.bs_y,
                            param.x_start*param.bs_x,(param.x_start+param.n_blocks_x)*param.bs_x]

    print("box_combined: " + str(box_combined), flush=True)

    output_folder = param.folder_path+"/z"+str(bz_global).zfill(4)+"_it_"+str(iteration)+"/"
    makeFolder(output_folder)
    dumpNumbaDictToFile(border_comp_combined_new, "border_comp_local", output_folder, "")
    dumpToFile(border_comp_exist_combined_new, "border_comp_exist_local", output_folder, "")
    dumpToFile(box_combined, "box_combined", output_folder, "")
    dumpToFile(neighbor_label_set_border_global, "neighbor_label_set_border_global", output_folder, "")

    del border_comp_combined, border_comp_exist_combined, border_comp_combined_new, border_comp_exist_combined_new, box_combined, neighbor_label_set_border_global

else:
    raise ValueError("Unknown Error")
