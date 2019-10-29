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

from functions import readFromFile, writeNeighborLabelDict, findAssociatedLabels, setUndeterminedtoNonHole, dumpNumbaDictToFile, blockFolderPath

start_time_total = time.time()

# final step
neighbor_label_set_border_global_combined = set()
associated_label_global = Dict.empty(key_type=types.int64,value_type=types.int64)
undetermined_global = set()
neighbor_label_dict = dict()

#timing
start_time_pickleload = time.time()

# load sets by iterating over all folders (expected to be small time consumption)
for bz in range(param.z_start, param.z_start+param.n_blocks_z):
    for by in range(param.y_start, param.y_start+param.n_blocks_y):
        for bx in range(param.x_start, param.x_start+param.n_blocks_x):

            output_folder = blockFolderPath(param.folder_path, bz,by,bx)

            neighbor_label_set_border_global = readFromFile("neighbor_label_set_border_global", output_folder, "")
            associated_label_local = readFromFile("associated_label_local", output_folder, "")
            undetermined_local = readFromFile("undetermined_local", output_folder, "")
            neighbor_label_dict_reduced_local = readFromFile("neighbor_label_dict_reduced", output_folder, "")

            if -2 in associated_label_local.keys():
                print("Assoc_label!")
                print(bz,by,bx)

            if -2 in undetermined_local:
                print("undet local!")
                print(bz,by,bx)

            neighbor_label_set_border_global_combined = neighbor_label_set_border_global_combined.union(neighbor_label_set_border_global)
            associated_label_global.update(associated_label_local)
            undetermined_global = undetermined_global.union(undetermined_local)
            neighbor_label_dict.update(neighbor_label_dict_reduced_local)

            del neighbor_label_set_border_global, associated_label_local, undetermined_local

len_label_set_border = len(neighbor_label_set_border_global_combined)

time_pickleload = time.time() - start_time_pickleload
start_time_findAssocLabelGlobal = time.time()

#compute associated label global
neighbor_label_dict = writeNeighborLabelDict(neighbor_label_dict=neighbor_label_dict, neighbor_label_set=neighbor_label_set_border_global_combined.copy())
associated_label_global, undetermined_global, isHole, isNotHole = findAssociatedLabels(neighbor_label_dict, undetermined_global, associated_label_global)
associated_label_global = setUndeterminedtoNonHole(undetermined_global.copy(), associated_label_global)

n_Holes = len(isHole)
n_NotHoles = len(isNotHole)+len(undetermined_global)

time_findAssocLabelGlobal = time.time() - start_time_findAssocLabelGlobal
start_time_splitandwritepickle = time.time()

# for bz in range(param.z_start, param.z_start+param.n_blocks_z):
#     for by in range(param.y_start, param.y_start+param.n_blocks_y):
#         for bx in range(param.x_start, param.x_start+param.n_blocks_x):
#
#             block_number = (bz)*(param.y_start+param.n_blocks_y)*(param.x_start+param.n_blocks_x)+by*(param.x_start+param.n_blocks_x)+bx
#             label_start = -1*block_number*param.max_labels_block-1
#             label_end = label_start - param.max_labels_block
#
#             associated_label_block = {key: value for key, value in associated_label_global.items() if (key>label_end and key<=label_start)}
#
#             output_folder = blockFolderPath(param.folder_path,bz,by,bx)
#             #save associated label global
#             output_name = ""
#             dumpNumbaDictToFile(associated_label_block, "associated_label_block", output_folder, output_name)

dumpNumbaDictToFile(associated_label_global, "associated_label_block", param.folder_path, "")

time_splitandwritepickle = time.time()-start_time_splitandwritepickle
time_total = time.time()-start_time_total

len_associated_label_global = len(associated_label_global)

g = open(param.step02B_info_filepath, "a+")
g.write(    "total_time," + format(time_total, '.4f') + "," +
            "pickleload_time," + format(time_pickleload, '.4f')+","+
            "findAssocLabelGlobal_time," + format(time_findAssocLabelGlobal, '.4f')+","+
            "splitandwritepickle_time," + format(time_splitandwritepickle, '.4f')+","+
            "len_label_set_border," + str(len_label_set_border).zfill(8)+","+
            "len_associated_label_global," + str(len_associated_label_global).zfill(8)+","+
            "n_Holes," + str(n_Holes).zfill(8)+","+
            "n_NotHoles," + str(n_NotHoles).zfill(8)+"\n")
g.close()
