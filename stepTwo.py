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


from functions import readFromFile, findAdjLabelSetGlobal, writeNeighborLabelDict, findAssociatedLabels, setUndeterminedtoNonHole, dumpNumbaDictToFile, makeFolder, dumpToFile, IdiToIdx

print("executing Step 2 calculations...", flush=True)

# STEP 2

neighbor_label_set_inside_global = set()
associated_label_global = Dict.empty(key_type=types.int64,value_type=types.int64)
undetermined_global = set()
neighbor_label_set_border_global = {(1,1)}

z_range = np.arange(param.z_start, param.z_start+param.n_blocks_z)

#iteration 1 (2 blokcs)
for bz_global in z_range[::2]:

    border_comp_combined = Dict.empty(key_type=types.int64,value_type=types.int64)
    border_comp_exist_combined = set()

    for bz in [bz_global, bz_global+1]:
        for by in range(param.y_start, param.y_start+param.n_blocks_y):
            for bx in range(param.x_start, param.x_start+param.n_blocks_x):

                print("Block z is: " + str(bz), flush=True)
                output_folder = param.folder_path+"/z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)+"/"

                border_comp_local = readFromFile("border_comp_local", output_folder, "")
                border_comp_exist_local = readFromFile("border_comp_exist_local", output_folder, "")
                neighbor_label_set_inside_local = readFromFile("neighbor_label_set_inside_local", output_folder, "")
                associated_label_local = readFromFile("associated_label_local", output_folder, "")
                undetermined_local = readFromFile("undetermined_local", output_folder, "")

                border_comp_combined.update(border_comp_local)
                border_comp_exist_combined = border_comp_exist_combined.union(border_comp_exist_local)

                neighbor_label_set_inside_global = neighbor_label_set_inside_global.union(neighbor_label_set_inside_local)
                associated_label_global.update(associated_label_local)
                undetermined_global = undetermined_global.union(undetermined_local)

                del border_comp_local, border_comp_exist_local, neighbor_label_set_inside_local, associated_label_local, undetermined_local

    for bz in [bz_global, bz_global+1]:

        if bz == bz_global:
            connectInPosZdirec = True
            connectInNegZdirec = False

        if bz == bz_global+1:
            connectInPosZdirec = False
            connectInNegZdirec = True

        for by in range(param.y_start, param.y_start+param.n_blocks_y):
            for bx in range(param.x_start, param.x_start+param.n_blocks_x):

                box = [bz*param.bs_z,(bz+1)*param.bs_z,by*param.bs_y,(by+1)*param.bs_y,bx*param.bs_x,(bx+1)*param.bs_x]
                print(box)

                # print(box)
                neighbor_label_set_border_global = findAdjLabelSetGlobal(box, neighbor_label_set_border_global,
                                                        border_comp_combined, border_comp_exist_combined, param.yres, param.xres, connectInPosZdirec, connectInNegZdirec)

    output_folder = param.folder_path+"/z"+str(bz_global).zfill(4)+"/"
    makeFolder(output_folder)
    dumpNumbaDictToFile(border_comp_combined, "border_comp_local", output_folder, "")
    dumpToFile(border_comp_exist_combined, "border_comp_exist_local", output_folder, "")

    del border_comp_combined, border_comp_exist_combined

#iteration2 (4 blocks)
for bz_global in z_range[::4]:

    border_comp_combined = Dict.empty(key_type=types.int64,value_type=types.int64)
    border_comp_exist_combined = set()

    for bz in [bz_global, bz_global+2]:

        print("Block z is: " + str(bz), flush=True)
        output_folder = param.folder_path+"/z"+str(bz).zfill(4)+"/"

        border_comp_local = readFromFile("border_comp_local", output_folder, "")
        border_comp_exist_local = readFromFile("border_comp_exist_local", output_folder, "")

        border_comp_combined.update(border_comp_local)
        border_comp_exist_combined = border_comp_exist_combined.union(border_comp_exist_local)

        del border_comp_local, border_comp_exist_local

    for bz in [bz_global, bz_global+2]:

        connectInPosZdirec = True # (this is the final block)
        connectInNegZdirec = True

        box = [bz*param.bs_z,(bz+2)*param.bs_z,param.y_start*param.bs_y,(param.y_start+param.n_blocks_y)*param.bs_y,param.x_start*param.bs_x,(param.x_start+param.n_blocks_x)*param.bs_x]

        print(box)

        # print(box)
        neighbor_label_set_border_global = findAdjLabelSetGlobal(box, neighbor_label_set_border_global,
                                                border_comp_combined, border_comp_exist_combined, param.yres, param.xres, connectInPosZdirec, connectInNegZdirec)

    del border_comp_combined, border_comp_exist_combined



print("Final: ")
print("neighbor_label_set_inside_global: " + str(sys.getsizeof(neighbor_label_set_inside_global)))
print("associated_label_global: " + str(sys.getsizeof(dict(associated_label_global))))
print("undetermined_global: " + str(sys.getsizeof(undetermined_global)))

print("Created border_comp_exist_global and neighbor_label_set_border_global", flush=True)

# counter_total = 0
#
# for bz in range(param.z_start, param.z_start+param.n_blocks_z):
#     for by in range(param.y_start, param.y_start+param.n_blocks_y):
#         for bx in range(param.x_start, param.x_start+param.n_blocks_x):
#
#             box = [bz*param.bs_z,(bz+1)*param.bs_z,by*param.bs_y,(by+1)*param.bs_y,bx*param.bs_x,(bx+1)*param.bs_x]
#             # print(box)
#
#             neighbor_label_set_border_global = findAdjLabelSetGlobal(box, neighbor_label_set_border_global,
#                                                     border_comp_global, border_comp_exist_global, param.yres, param.xres)

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
