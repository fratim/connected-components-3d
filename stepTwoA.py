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


from functions import readFromFile, findAdjLabelSetGlobal, makeFolder, dumpToFile, blockFolderPath

if(len(sys.argv))!=4:
    raise ValueError(" Scripts needs exactley 3 input arguments (bz, vy, bx)")
else:
    bz = int(sys.argv[1])
    by = int(sys.argv[2])
    bx = int(sys.argv[3])

# timing
start_time_total = time.time()

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

start_time_AdjLabelGlobal = time.time()
neighbor_label_set_border_global = findAdjLabelSetGlobal(param.folder_path,neighbor_label_set_border_global,border_contact,bz,by,bx)
time_AdjLabelGlobal = time.time() - start_time_AdjLabelGlobal

neighbor_label_set_border_global.remove((1,1))

output_folder = blockFolderPath(param.folder_path,bz,by,bx)

start_time_picklewrite = time.time()
dumpToFile(neighbor_label_set_border_global, "neighbor_label_set_border_global", output_folder, "")
time_picklewrite = time.time() - start_time_picklewrite

time_total = time.time()-start_time_total

g = open(param.step02A_info_filepath, "a+")
g.write(    "bz/by/bx,"+str(bz).zfill(4)+","+str(by).zfill(4)+","+str(bx).zfill(4)+","
            "total_time," + format(time_total, '.4f')+ "," +
            "AdjLabelGlobal_time," + format(time_AdjLabelGlobal, '.4f')+","+
            "picklewrite_time," + format(time_picklewrite, '.4f')+"\n")
g.close()

if param.compute_statistics:
    g = open(param.component_equivalences_filepath, "a+")
    for entry in neighbor_label_set_border_global.keys():
        g.write(str(int(entry[0])).zfill(25)+", "+str(int(entry[1])).zfill(25)+"\n")
    g.close()
