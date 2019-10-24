import cc3d
import numpy as np
import time
import h5py
from numba import njit, types
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from numba.typed import Dict
import os
import sys
import pickle
import param

from functions import makeFolder, dataBlock

# set will be deprecated soon on numba, but until now an alternative has not been implemented
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# pass arguments
if(len(sys.argv))!=4:
    raise ValueError(" Scripts needs exactley 3 input arguments (bz by bx)")
else:
    bz = int(sys.argv[1])
    by = int(sys.argv[2])
    bx = int(sys.argv[3])

# compute and save variables and data
start_time_read_labels = time.time()
start_time_total = time.time()

block_number = (bz)*(param.y_start+param.n_blocks_y)*(param.x_start+param.n_blocks_x)+by*(param.x_start+param.n_blocks_x)+bx
label_start = -1*block_number*param.max_labels_block

currBlock = dataBlock(viz_wholes=True)
currBlock.readLabels(data_path=param.data_path, sample_name=param.sample_name,
                        bz=bz, by=by, bx=bx)

time_read_labels = time.time()-start_time_read_labels

currBlock.setRes(zres=param.zres,yres=param.yres,xres=param.xres)

currBlock.computeStepOne(label_start=label_start, max_labels_block=param.max_labels_block, output_path=param.folder_path)


time_total = time.time()-start_time_total
# write n_comp to text file that all blocks write to
f = open(param.n_comp_filepath, "a+")
f.write(str(block_number).zfill(8)+","+str(currBlock.n_comp).zfill(12)+"\n")
f.close()

g = open(param.step01_timing_filepath, "a+")
g.write(    "total," + format(time_total, '.4f') + "," +
            "readLabels," + format(time_read_labels, '.4f')+","+
            "ccLabels," + format(currBlock.time_cc_labels, '.4f')+","+
            "asjLabelLocal," + format(currBlock.time_AdjLabelLocal, '.4f')+","+
            "assocLabel," + format(currBlock.time_assoc_labels, '.4f')+","+
            "pickle," + format(currBlock.time_writepickle, '.4f')+"\n")
g.close()
