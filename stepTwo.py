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


from functions import readFromFile, findAdjLabelSetGlobal, writeNeighborLabelDict
from functions import findAssociatedLabels, setUndeterminedtoNonHole, dumpNumbaDictToFile, dumpToFile, blockFolderPath

print("executing Step 2 calculations...", flush=True)

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

for bz in range(param.z_start, param.z_start+param.n_blocks_z):
    for by in range(param.y_start, param.y_start+param.n_blocks_y):
        for bx in range(param.x_start, param.x_start+param.n_blocks_x):

            block_number = (bz)*(param.y_start+param.n_blocks_y)*(param.x_start+param.n_blocks_x)+by*(param.x_start+param.n_blocks_x)+bx
            label_start = -1*block_number*param.max_labels_block-1
            label_end = label_start - param.max_labels_block

            associated_label_block = {key: value for key, value in associated_label_global.items() if (key>label_end and key<=label_start)}

            output_folder = blockFolderPath(param.folder_path,bz,by,bx)
            #save associated label global
            output_name = ""
            dumpNumbaDictToFile(associated_label_block, "associated_label_block", output_folder, output_name)
