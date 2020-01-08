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
import cc3d
import numpy as np

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

@njit
def evaluateLabels(cc_labels, labels_in, n_comp):

    items_of_component = Dict.empty(key_type=types.int64,value_type=types.int64)
    label_to_cclabel = Dict.empty(key_type=types.int64,value_type=float_array)
    cc_labels_known = set()
    label_to_cclabel_keys = set()

    for j in range(n_comp):
        items_of_component[j]=0

    for iz in range(cc_labels.shape[0]):
        for iy in range(cc_labels.shape[1]):
            for ix in range(cc_labels.shape[2]):

                curr_comp = cc_labels[iz,iy,ix]
                if curr_comp!=0:
                    items_of_component[curr_comp]+=1
                    if curr_comp not in cc_labels_known:
                        cc_labels_known.add(curr_comp)
                        if labels_in[iz,iy,ix] in label_to_cclabel_keys:
                            add = np.array([curr_comp]).astype(np.int64)
                            label_to_cclabel[labels_in[iz,iy,ix]] = np.concatenate((label_to_cclabel[labels_in[iz,iy,ix]].ravel(), add))
                        else:
                            label_to_cclabel[labels_in[iz,iy,ix]] = np.array([curr_comp],dtype=np.int64).astype(np.int64)
                            label_to_cclabel_keys.add(labels_in[iz,iy,ix])

    return items_of_component, label_to_cclabel

@njit
def update_labels(keep_labels, labels_in, cc_labels):
    for iz in range(labels_in.shape[0]):
        for iy in range(labels_in.shape[1]):
            for ix in range(labels_in.shape[2]):

                if cc_labels[iz,iy,ix] not in keep_labels:
                    labels_in[iz,iy,ix]=0

    return labels_in

# compute and save variables and data
block_number = (bz)*(param.y_start+param.n_blocks_y)*(param.x_start+param.n_blocks_x)+by*(param.x_start+param.n_blocks_x)+bx
label_start = -1*block_number*param.max_labels_block

currBlock = dataBlock(viz_wholes=True)
currBlock.readLabels(data_path=param.data_path, sample_name=param.sample_name,
                        bz=bz, by=by, bx=bx)

cc_labels = cc3d.connected_components(currBlock.labels_in, connectivity=26)

n_comp = np.max(cc_labels) + 1

items_of_component, label_to_cclabel = evaluateLabels(cc_labels, currBlock.labels_in, n_comp)

keep_labels = set()

for entry in label_to_cclabel.keys():

    print(entry)

    most_points = -1
    largest_comp = -1
    for comp in label_to_cclabel[entry]:
        # print(items_of_component[comp])
        if items_of_component[comp]>most_points:
            largest_comp = comp
            most_points = items_of_component[comp]

    keep_labels.add(largest_comp)

currBlock.labels_in = update_labels(keep_labels, currBlock.labels_in, cc_labels)
