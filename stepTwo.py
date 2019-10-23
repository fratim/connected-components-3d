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


from functions import readFromFile, findAdjLabelSetGlobal, writeNeighborLabelDict, findAssociatedLabels, setUndeterminedtoNonHole, dumpNumbaDictToFile, makeFolder, dumpToFile, IdiToIdx, combineBoxes

print("executing Step 2 calculations...", flush=True)

# STEP 2

z_range = np.arange(param.z_start, param.z_start+param.n_blocks_z)


# iteration 1 (2 blocks)
for bz_global in z_range:

    border_comp_combined = Dict.empty(key_type=types.int64,value_type=types.int64)
    border_comp_exist_combined = set()
    neighbor_label_set_border_global = {(1,1)}

    # determine if Block is border block
    if bz_global+1<=z_range[-1] and bz_global-1>=z_range[0]:
        bz_range = [bz_global -1, bz_global, bz_global+1]
        isBorder = False
    elif bz_global+1<=z_range[-1] and bz_global-1<z_range[0]:
        bz_range = [bz_global, bz_global+1]
        hasBorderLeft = True
        hasBorderRight = False
    elif bz_global+1>z_range[-1] and bz_global-1>=z_range[0]:
        bz_range = [bz_global-1, bz_global]
        hasBorderLeft = False
        hasBorderRight = True
    elif bz_global+1>z_range[-1] and bz_global-1<z_range[0]:
        bz_range = [bz_global-1, bz_global]
        hasBorderLeft = True
        hasBorderRight = True
    else:
        raise ValueError("Unknown Error!")

    print(bz_range)

    for bz in bz_range:
        for by in range(param.y_start, param.y_start+param.n_blocks_y):
            for bx in range(param.x_start, param.x_start+param.n_blocks_x):

                output_folder = param.folder_path+"/z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)+"/"

                border_comp_local = readFromFile("border_comp_local", output_folder, "")
                border_comp_exist_local = readFromFile("border_comp_exist_local", output_folder, "")

                border_comp_combined.update(border_comp_local)
                border_comp_exist_combined = border_comp_exist_combined.union(border_comp_exist_local)

                del border_comp_local, border_comp_exist_local

    # TODO should be able to delete this
    border_comp_combined_new = Dict.empty(key_type=types.int64,value_type=types.int64)
    border_comp_exist_combined_new = {(2**30)}

    connectInPosZdirec = True
    connectInNegZdirec = True

    for by in range(param.y_start, param.y_start+param.n_blocks_y):
        for bx in range(param.x_start, param.x_start+param.n_blocks_x):

            # find box to iterative over all blocks
            box = [bz_global*param.bs_z,(bz_global+1)*param.bs_z,by*param.bs_y,(by+1)*param.bs_y,bx*param.bs_x,(bx+1)*param.bs_x]
            print(box)
            _, _, neighbor_label_set_border_global = findAdjLabelSetGlobal(box, neighbor_label_set_border_global,
                                                    border_comp_combined, border_comp_exist_combined, param.yres, param.xres, connectInPosZdirec, connectInNegZdirec,
                                                    border_comp_combined_new, border_comp_exist_combined_new)

    border_comp_exist_combined_new.remove((2**30))
    neighbor_label_set_border_global.remove((1,1))

    output_folder = param.folder_path+"/z"+str(bz_global).zfill(4)+"/"
    makeFolder(output_folder)
    dumpToFile(neighbor_label_set_border_global, "neighbor_label_set_border_global", output_folder, "")

    del border_comp_combined, border_comp_exist_combined, border_comp_combined_new, border_comp_exist_combined_new, neighbor_label_set_border_global


# final step
neighbor_label_set_inside_global = set()
associated_label_global = Dict.empty(key_type=types.int64,value_type=types.int64)
undetermined_global = set()
neighbor_label_set_border_global_combined = set()

# load sets by iterating over all folders (expected to be small)
for bz in range(param.z_start, param.z_start+param.n_blocks_z):
    for by in range(param.y_start, param.y_start+param.n_blocks_y):
        for bx in range(param.x_start, param.x_start+param.n_blocks_x):

            output_folder = param.folder_path+"/z"+str(bz).zfill(4)+"/"
            neighbor_label_set_border_global = readFromFile("neighbor_label_set_border_global", output_folder, "")
            neighbor_label_set_border_global_combined = neighbor_label_set_border_global_combined.union(neighbor_label_set_border_global)

            output_folder = param.folder_path+"/z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)+"/"
            neighbor_label_set_inside_local = readFromFile("neighbor_label_set_inside_local", output_folder, "")
            associated_label_local = readFromFile("associated_label_local", output_folder, "")
            undetermined_local = readFromFile("undetermined_local", output_folder, "")
            neighbor_label_set_inside_global = neighbor_label_set_inside_global.union(neighbor_label_set_inside_local)
            associated_label_global.update(associated_label_local)
            undetermined_global = undetermined_global.union(undetermined_local)

            del neighbor_label_set_border_global, neighbor_label_set_inside_local, associated_label_local, undetermined_local

#unify label set inside and outside
neighbor_label_set = neighbor_label_set_inside_global.union(neighbor_label_set_border_global_combined)

#compute associated label global
neighbor_label_dict = writeNeighborLabelDict(neighbor_label_set)
associated_label_global, undetermined_global = findAssociatedLabels(neighbor_label_dict, undetermined_global, associated_label_global)
associated_label_global = setUndeterminedtoNonHole(undetermined_global, associated_label_global)

#save associated label global
output_name = ""
dumpNumbaDictToFile(associated_label_global, "associated_label_global", param.folder_path, output_name)
