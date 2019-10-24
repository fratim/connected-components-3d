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
y_range = np.arange(param.y_start, param.y_start+param.n_blocks_y)
x_range = np.arange(param.x_start, param.x_start+param.n_blocks_x)


# iteration 1 (2 blocks)
for bz in range(param.z_start, param.z_start+param.n_blocks_z):
    for by in range(param.y_start, param.y_start+param.n_blocks_y):
        for bx in range(param.x_start, param.x_start+param.n_blocks_x):

            if bz!=z_range[-1]:
                bs_z = param.bs_z
                bs_y = param.bs_y
                bs_x = param.bs_x
            elif bz==z_range[-1]:
                bs_z = param.bs_z_last
                bs_y = param.bs_y
                bs_x = param.bs_x
            else: raise ValueError("Unknown Error")

            # determine if Block is border block Z direc
            if bz<(param.z_start+param.n_blocks_z-1) and bz>param.z_start:
                hasZBorder = False
                onZMinBorder =  False
                onZMaxBorder =  False
            elif bz<(param.z_start+param.n_blocks_z-1) and bz==param.z_start:
                hasZBorder = True
                onZMinBorder = True
                onZMaxBorder = False
            elif bz==(param.z_start+param.n_blocks_z-1) and bz>param.z_start:
                hasZBorder = True
                onZMinBorder = False
                onZMaxBorder = True
            elif bz==(param.z_start+param.n_blocks_z-1) and bz==param.z_start:
                hasZBorder = True
                onZMinBorder = True
                onZMaxBorder = True
            else:
                raise ValueError("Unknown Error Z!")

            # determine if Block is border block Y dirc
            if by<(param.y_start+param.n_blocks_y-1) and by>param.y_start:
                hasYBorder = False
                onYMinBorder = False
                onYMaxBorder = False
            elif by<(param.y_start+param.n_blocks_y-1) and by==param.y_start:
                hasYBorder = True
                onYMinBorder = True
                onYMaxBorder = False
            elif by==(param.y_start+param.n_blocks_y-1) and by>param.y_start:
                hasYBorder = True
                onYMinBorder = False
                onYMaxBorder = True
            elif by==(param.y_start+param.n_blocks_y-1) and by==param.y_start:
                hasYBorder = True
                onYMinBorder = True
                onYMaxBorder = True
            else:
                raise ValueError("Unknown Error Y!")

            # determine if Block is border block X direc
            if bx<(param.x_start+param.n_blocks_x-1) and bx>param.x_start:
                hasXBorder = False
                onXMinBorder = False
                onXMaxBorder = False
            elif bx<(param.x_start+param.n_blocks_x-1) and bx==param.x_start:
                hasXBorder = True
                onXMinBorder = True
                onXMaxBorder = False
            elif bx==(param.x_start+param.n_blocks_x-1) and bx>param.x_start:
                hasXBorder = True
                onXMinBorder = False
                onXMaxBorder = True
            elif bx==(param.x_start+param.n_blocks_x-1) and bx==param.x_start:
                hasXBorder = True
                onXMinBorder = True
                onXMaxBorder = True
            else:
                raise ValueError("Unknown Error X!")

            neighbor_label_set_border_global = {(1,1)}

            border_contact = [onZMinBorder, onZMaxBorder, onYMinBorder, onYMaxBorder, onXMinBorder, onXMaxBorder]

            box = [bz*bs_z,(bz+1)*bs_z,by*bs_y,(by+1)*bs_y,bx*bs_x,(bx+1)*bs_x]

            neighbor_label_set_border_global = findAdjLabelSetGlobal(box,param.folder_path,neighbor_label_set_border_global,param.yres,param.xres,border_contact,bz,by,bx)

            output_folder = param.folder_path+"/z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)+"/"
            dumpToFile(neighbor_label_set_border_global, "neighbor_label_set_border_global", output_folder, "")

# final step
neighbor_label_set_inside_global = set()
associated_label_global = Dict.empty(key_type=types.int64,value_type=types.int64)
undetermined_global = set()
neighbor_label_set_border_global_combined = set()

# load sets by iterating over all folders (expected to be small)
for bz in range(param.z_start, param.z_start+param.n_blocks_z):
    for by in range(param.y_start, param.y_start+param.n_blocks_y):
        for bx in range(param.x_start, param.x_start+param.n_blocks_x):

            output_folder = param.folder_path+"/z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)+"/"
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
