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

from functions import fillWholes, readFromFile

# pass arguments
if(len(sys.argv))!=4:
    raise ValueError(" Scripts needs exactley 3 input arguments (bz by bx)")
else:
    bz = int(sys.argv[1])
    by = int(sys.argv[2])
    bx = int(sys.argv[3])

# STEP 3
output_name = ""
associated_label_global = Dict.empty(key_type=types.int64,value_type=types.int64)
associated_label_global.update(readFromFile("associated_label_global", param.folder_path, output_name))

output_folder = param.folder_path+"/z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)+"/"
fillWholes(output_path=output_folder,bz=bz,by=by,bx=bx,associated_label=associated_label_global)
