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


from functions import readFromFile, findAdjLabelSetGlobal, makeFolder, dumpToFile

if(len(sys.argv))!=2:
    raise ValueError(" Scripts needs exactley 3 input arguments (bz)")
else:
    bz_global = int(sys.argv[1])

if bz_global!=z_range[-1]:
    bs_z = param.bs_z
    bs_y = param.bs_y
    bs_x = param.bs_x
elif bz_global==z_range[-1]:
    bs_z = param.bs_z_last
    bs_y = param.bs_y
    bs_x = param.bs_x
else: raise ValueError("Unknown Error")

print("executing Step 2 calculations block " + str(bz_global), flush=True)

# STEP 2

z_range = np.arange(param.z_start, param.z_start+param.n_blocks_z)


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

connectInPosZdirec = True
connectInNegZdirec = True

for by in range(param.y_start, param.y_start+param.n_blocks_y):
    for bx in range(param.x_start, param.x_start+param.n_blocks_x):

        # find box to iterative over all blocks
        box = [bz_global*bs_z,(bz_global+1)*bs_z,by*bs_y,(by+1)*bs_y,bx*bs_x,(bx+1)*bs_x]
        print(box)
        neighbor_label_set_border_global = findAdjLabelSetGlobal(box, neighbor_label_set_border_global,
                                                border_comp_combined, border_comp_exist_combined, param.yres, param.xres, connectInPosZdirec, connectInNegZdirec)

output_folder = param.folder_path+"/z"+str(bz_global).zfill(4)+"/"
makeFolder(output_folder)
dumpToFile(neighbor_label_set_border_global, "neighbor_label_set_border_global", output_folder, "")

del border_comp_combined, border_comp_exist_combined, neighbor_label_set_border_global
