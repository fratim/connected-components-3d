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


from functions import readFromFile, findAdjLabelSetGlobal, dumpNumbaDictToFile, makeFolder, dumpToFile, combineBoxes

# pass arguments
if(len(sys.argv))!=3:
    raise ValueError(" Scripts needs exactley 2 input arguments (bz, iteration)")
else:
    bz_global = int(sys.argv[1])
    iteration = int(sys.argv[2])

# Iteration 1 interconnects 2 z-blocks, also interconnecting in all x-y directions and loading the needed variables
iterations_needed = param.iterations_needed
z_range = np.arange(param.z_start, param.z_start+param.n_blocks_z)

print("------------------------------------------------", flush=True)
print("Iteration " + str(iteration), flush=True)
print("bz_global is: " + str(bz_global), flush=True)

block_size = 2**iteration

if bz_global not in z_range[::block_size]:
    print("Block not computed in this iteration, aborted. ", flush=True)

elif bz_global in z_range[::block_size]:

    print(str(bz_global) + " is in z_range")

    #doublecheck
    if z_range[0]+block_size > z_range[-1] and iteration != iterations_needed:
          raise ValueError("Unknown Error")

    border_comp_combined = Dict.empty(key_type=types.int64,value_type=types.int64)
    border_comp_exist_combined = set()
    neighbor_label_set_border_global_combined = set()

    # check if block is missing and have to compute single
    if bz_global+int(block_size/2)>z_range[-1]:
        bz_range = [bz_global]
        isSingle = True
    elif bz_global+int(block_size/2)<=z_range[-1]:
        bz_range = [bz_global, bz_global+int(block_size/2)]
        isSingle=False
    else:
        raise ValueError("Unknown Error!")

    for bz in bz_range:

        output_folder = param.folder_path+"/z"+str(bz).zfill(4)+"_it_"+str(iteration-1)+"/"

        # load coordinate boxes
        if bz==bz_range[0]:
            box_a = readFromFile("box_combined", output_folder, "")
        elif bz==bz_range[1]:
            box_b = readFromFile("box_combined", output_folder, "")
        else:
            raise ValueError("Unknown Error")

        #load border components from last iteration
        border_comp_local = readFromFile("border_comp_local", output_folder, "")
        border_comp_exist_local = readFromFile("border_comp_exist_local", output_folder, "")
        neighbor_label_set_border_global = readFromFile("neighbor_label_set_border_global", output_folder, "")
        border_comp_combined.update(border_comp_local)
        border_comp_exist_combined = border_comp_exist_combined.union(border_comp_exist_local)
        neighbor_label_set_border_global_combined = neighbor_label_set_border_global_combined.union(neighbor_label_set_border_global)
        del border_comp_local, border_comp_exist_local, neighbor_label_set_border_global

    print("------------------")
    print("bz_range: " + str(bz_range))
    if isSingle:
        box_combined = box_a
        print("box_a       : " + str(box_a))
    if not isSingle:
        box_combined = combineBoxes(box_a, box_b)
        print("box_a:        " + str(box_a))
        print("box_b:        " + str(box_b))
    print("box_combined: " + str(box_combined))


    border_comp_combined_new = Dict.empty(key_type=types.int64,value_type=types.int64)
    border_comp_exist_combined_new = {(2**30)}

    for bz in bz_range:

        # set z directions to consider
        if iteration == iterations_needed:
            connectInPosZdirec = True
            connectInNegZdirec = True
        elif isSingle:
            connectInPosZdirec = False
            connectInNegZdirec = False
        else:
            if bz == bz_range[0]:
                connectInPosZdirec = True
                connectInNegZdirec = False
            elif bz == bz_range[1]:
                connectInPosZdirec = False
                connectInNegZdirec = True
            else:
                raise ValueError("Error in determining z connection directions")

        # load correct coordinate box
        if bz == bz_range[0]:
            box = box_a
        elif not isSingle and bz==bz_range[1]:
            box = box_b
        else:
            raise ValueError("Unknown Error")

        # associated label global and border components for next iteration
        border_comp_combined_new, border_comp_exist_combined_new, neighbor_label_set_border_global = findAdjLabelSetGlobal(box, neighbor_label_set_border_global_combined,
                                                border_comp_combined, border_comp_exist_combined, param.yres, param.xres, connectInPosZdirec, connectInNegZdirec,
                                                border_comp_combined_new, border_comp_exist_combined_new)

    border_comp_exist_combined_new.remove((2**30))

    output_folder = param.folder_path+"/z"+str(bz_global).zfill(4)+"_it_"+str(iteration)+"/"
    makeFolder(output_folder)
    dumpNumbaDictToFile(border_comp_combined_new, "border_comp_local", output_folder, "")
    dumpToFile(border_comp_exist_combined_new, "border_comp_exist_local", output_folder, "")
    dumpToFile(box_combined, "box_combined", output_folder, "")
    dumpToFile(neighbor_label_set_border_global, "neighbor_label_set_border_global", output_folder, "")

    del border_comp_combined, border_comp_exist_combined, border_comp_combined_new
    del border_comp_exist_combined_new, box_combined, neighbor_label_set_border_global, neighbor_label_set_border_global_combined

else:
    raise ValueError("Unknown Error")
