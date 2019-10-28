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

currBlock.computeStepOne(label_start=label_start, output_path=param.folder_path)

time_total = time.time()-start_time_total

# write info parameters to file
g = open(param.step01_info_filepath, "a+")
g.write(    "bz/by/bx,"+str(bz).zfill(4)+","+str(by).zfill(4)+","+str(bx).zfill(4)+","
            "total_time," + format(time_total, '.4f') + "," +
            "readLabels_time," + format(time_read_labels, '.4f')+","+
            "ccLabels_time," + format(currBlock.time_cc_labels, '.4f')+","+
            "adjLabelLocal_time," + format(currBlock.time_AdjLabelLocal, '.4f')+","+
            "assocLabel_time," + format(currBlock.time_assoc_labels, '.4f')+","+
            "pickle_time," + format(currBlock.time_writepickle, '.4f')+","+
            "n_comp," + str(currBlock.n_comp).zfill(12)+","+
            "n_Holes," + str(currBlock.n_Holes).zfill(8)+","+
            "n_NotHoles," + str(currBlock.n_NotHoles).zfill(8)+","+
            "len_label_set_inside," + str(currBlock.size_label_set_inside).zfill(8)+","+
            "len_label_set_inside_reduced," + str(currBlock.size_label_set_inside_reduced).zfill(8)+"\n")
g.close()
