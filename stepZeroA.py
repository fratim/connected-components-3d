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

from functions import makeFolder, dataBlock, readData, writeData

# set will be deprecated soon on numba, but until now an alternative has not been implemented
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

float_array = types.int64[:]
dsp_factor = 8

@njit
def evaluateLabels(cc_labels, labels_in, n_comp):

    point_of_component = Dict.empty(key_type=types.int64,value_type=float_array)
    items_of_component = Dict.empty(key_type=types.int64,value_type=types.int64)
    label_to_cclabel = Dict.empty(key_type=types.int64,value_type=float_array)
    cc_labels_known = set()
    cc_labels_known_block = set()
    label_to_cclabel_keys = set()

    for j in range(n_comp):
        items_of_component[j]=0

    reset_counter = 0

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

                    if curr_comp not in cc_labels_known_block:
                        point_of_component[curr_comp] = np.array([iz*dsp_factor,iy*dsp_factor,ix*dsp_factor],dtype=np.int64).astype(np.int64)
                        cc_labels_known_block.add(curr_comp)

                if ix == (param.max_bs_x-1):
                    cc_labels_known_block = set()
                    reset_counter += 1

            if iy == (param.max_bs_y-1):
                cc_labels_known_block = set()
                reset_counter += 1

        if iz == (param.max_bs_z-1):
            cc_labels_known_block = set()
            reset_counter += 1

    print("Blocks: " + str(reset_counter))

    return items_of_component, label_to_cclabel, point_of_component

@njit
def update_labels(keep_labels, labels_in, cc_labels):
    for iz in range(labels_in.shape[0]):
        for iy in range(labels_in.shape[1]):
            for ix in range(labels_in.shape[2]):

                if cc_labels[iz,iy,ix] not in keep_labels:
                    labels_in[iz,iy,ix]=0

    return labels_in

# filename = param.data_path+"/"+param.sample_name+"/"+"Zebrafinch-input_labels-"+str(bz).zfill(4)+"z-"+str(by).zfill(4)+"y-"+str(bx).zfill(4)+"x"
filename = segmentation_dsp8/Zebrafinch/Zebrafinch-seg-dsp_8"
box = [1]
labels_in = readData(box, filename)

cc_labels = cc3d.connected_components(labels_in, connectivity=26)

n_comp = np.max(cc_labels) + 1

items_of_component, label_to_cclabel, point_of_component = evaluateLabels(cc_labels, labels_in, n_comp)

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

g = open("segmentation_dsp8/Zebrafinch/component_anchors.txt", "w+")
for entry in keep_labels:
    g.write(str(int(point_of_component[entry][2])).zfill(10)+" "+ str(int(point_of_component[entry][1])).zfill(10)+" "+str(int(point_of_component[entry][0])).zfill(10)+"\n")
g.close()

# labels_in = update_labels(keep_labels, labels_in, cc_labels)

# filename = param.data_path+"/"+param.sample_name+"/"+"Zebrafinch-input_labels_discarded-"+str(bz).zfill(4)+"z-"+str(by).zfill(4)+"y-"+str(bx).zfill(4)+"x"

# writeData(filename, labels_in)
