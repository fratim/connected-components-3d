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


from functions import readFromFile, findAdjLabelSetGlobal, writeNeighborLabelDict, findAssociatedLabels, setUndeterminedtoNonHole, dumpNumbaDictToFile

print("executing Step 2 calculations...", flush=True)

# STEP 2
border_comp_global = Dict.empty(key_type=types.int64,value_type=types.int64)
border_comp_exist_global = {(2**30)}
neighbor_label_set_inside_global = set()
associated_label_global = Dict.empty(key_type=types.int64,value_type=types.int64)
undetermined_global = set()

for bz in range(param.z_start, param.z_start+param.n_blocks_z):
    for by in range(param.y_start, param.y_start+param.n_blocks_y):
        for bx in range(param.x_start, param.x_start+param.n_blocks_x):

            print("Block z is: " + str(bz), flush=True)
            output_folder = param.folder_path+"/z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)+"/"

            border_comp_local = readFromFile("border_comp_local", output_folder, "")
            border_comp_exist_local = readFromFile("border_comp_exist_local", output_folder, "")
            neighbor_label_set_inside_local = readFromFile("neighbor_label_set_inside_local", output_folder, "")
            associated_label_local = readFromFile("associated_label_local", output_folder, "")
            undetermined_local = readFromFile("undetermined_local", output_folder, "")

            # print("Border_comp_global: " + str(sys.getsizeof(border_comp_global)))
            # print("Border_com_exist_global: " + str(sys.getsizeof(border_comp_exist_global)))
            # print("neighbor_label_set_inside_global: " + str(sys.getsizeof(neighbor_label_set_inside_global)))
            # print("associated_label_global: " + str(sys.getsizeof(associated_label_global)))
            # print("undetermined_global: " + str(sys.getsizeof(undetermined_global)))

            border_comp_global.update(border_comp_local)
            border_comp_exist_global = border_comp_exist_global.union(border_comp_exist_local)
            neighbor_label_set_inside_global = neighbor_label_set_inside_global.union(neighbor_label_set_inside_local)
            associated_label_global.update(associated_label_local)
            undetermined_global = undetermined_global.union(undetermined_local)

            del border_comp_local, border_comp_exist_local, neighbor_label_set_inside_local, associated_label_local, undetermined_local

border_comp_exist_global.remove((2**30))
neighbor_label_set_border_global = {(1,1)}

print("Final: ")
print("Border_comp_global: " + str(sys.getsizeof(dict(border_comp_global))))
print("Border_com_exist_global: " + str(sys.getsizeof(border_comp_exist_global)))
print("neighbor_label_set_inside_global: " + str(sys.getsizeof(neighbor_label_set_inside_global)))
print("associated_label_global: " + str(sys.getsizeof(dict(associated_label_global))))
print("undetermined_global: " + str(sys.getsizeof(undetermined_global)))

print("Created border_comp_exist_global and neighbor_label_set_border_global", flush=True)

counter_total = 0

border_comp_combined_new = Dict.empty(key_type=types.int64,value_type=types.int64)
border_comp_exist_combined_new = {(2**30)}

for bz in range(param.z_start, param.z_start+param.n_blocks_z):
    for by in range(param.y_start, param.y_start+param.n_blocks_y):
        for bx in range(param.x_start, param.x_start+param.n_blocks_x):

            box = [bz*param.bs_z,(bz+1)*param.bs_z,by*param.bs_y,(by+1)*param.bs_y,bx*param.bs_x,(bx+1)*param.bs_x]
            # print(box)
            _, _, neighbor_label_set_border_global = findAdjLabelSetGlobal(box, neighbor_label_set_border_global,
                                                    border_comp_global, border_comp_exist_global, param.yres, param.xres, True, True,
                                                    border_comp_combined_new, border_comp_exist_combined_new)

neighbor_label_set_border_global.remove((1,1))
neighbor_label_set = neighbor_label_set_inside_global.union(neighbor_label_set_border_global)

print("Find associated labels...", flush=True)
neighbor_label_dict = writeNeighborLabelDict(neighbor_label_set)
associated_label_global, undetermined_global = findAssociatedLabels(neighbor_label_dict, undetermined_global, associated_label_global)
associated_label_global = setUndeterminedtoNonHole(undetermined_global, associated_label_global)

print("Final: ")
print("neighbor_label_dict: " + str(sys.getsizeof(neighbor_label_dict)))
print("neighbor_label_set: " + str(sys.getsizeof(neighbor_label_set)))
print("associated_label_global: " + str(sys.getsizeof(dict(associated_label_global))))
print("undetermined_global: " + str(sys.getsizeof(undetermined_global)))

output_name = ""
dumpNumbaDictToFile(associated_label_global, "associated_label_global", param.folder_path, output_name)