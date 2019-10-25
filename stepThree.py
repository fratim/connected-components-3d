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

from functions import fillWholes, readFromFile, readData, blockFolderPath

# pass arguments
if(len(sys.argv))!=4:
    raise ValueError(" Scripts needs exactley 3 input arguments (bz by bx)")
else:
    bz = int(sys.argv[1])
    by = int(sys.argv[2])
    bx = int(sys.argv[3])


# STEP 3

start_time_total = time.time()

start_time_readpickle = time.time()
output_folder = blockFolderPath(param.folder_path,bz,by,bx)
output_name = ""
associated_label_block = Dict.empty(key_type=types.int64,value_type=types.int64)
associated_label_block.update(readFromFile("associated_label_block", output_folder, output_name))
time_readpickle = time.time()-start_time_readpickle


start_time_fillWholes = time.time()
output_folder = param.folder_path+"/z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)+"/"
fillWholes(output_path=output_folder,associated_label=associated_label_block, bz=bz)
time_fillWholes = time.time() - start_time_fillWholes

time_total = time.time()-start_time_total

g = open(param.step03_timing_filepath, "a+")
g.write(    "total," + format(time_total, '.4f') + "," +
            "pickleload," + format(time_readpickle, '.4f')+","+
            "fillWholes," + format(time_fillWholes, '.4f')+"\n")
g.close()
