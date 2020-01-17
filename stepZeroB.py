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

# pass arguments
if(len(sys.argv))!=4:
    raise ValueError(" Scripts needs exactley 3 input arguments (bz by bx)")
else:
    bz = int(sys.argv[1])
    by = int(sys.argv[2])
    bx = int(sys.argv[3])

@njit
def update_labels(keep_labels, labels_in, cc_labels):
    
    for iz in range(labels_in.shape[0]):
        for iy in range(labels_in.shape[1]):
            for ix in range(labels_in.shape[2]):

                if cc_labels[iz,iy,ix] not in keep_labels:
                    labels_in[iz,iy,ix]=0

    return labels_in

filename = param.data_path+"/"+param.sample_name+"/"+"Zebrafinch-input_labels-"+str(bz).zfill(4)+"z-"+str(by).zfill(4)+"y-"+str(bx).zfill(4)+"x"
box = [1]
labels_in = readData(box, filename)

cc_labels = cc3d.connected_components(labels_in, connectivity=26)

filename_anchors = param.data_path+"/"+param.sample_name+"/"+"segmentation_dsp8/Zebrafinch/component_anchors.txt"

bs_z = labels_in.shape[0]
bs_y = labels_in.shape[1]
bs_x = labels_in.shape[2]

print("Blocksize:")
print(bs_z,bs_y,bs_x)

keep_labels = set()

with open(filename_anchors, 'r') as fd:
    for point in fd:
        # remove spacing
        point = point.strip().split()

        point_ix_gb = int(point[0])
        point_iy_gb = int(point[1])
        point_iz_gb = int(point[2])

        point_ix_local = point_ix_gb - bs_x*bx
        point_iy_local = point_iy_gb - bs_y*by
        point_iz_local = point_iz_gb - bs_z*bz
        
        if point_iz_local>=0 and point_iz_local<bs_z:
            if point_iy_local>=0 and point_iy_local<bs_y:
                if point_ix_local>=0 and point_ix_local<bs_x:
                    keep_labels.add(cc_labels[point_iz_local,point_iy_local,point_ix_local])
                    print("point added")
                    print("global, local")
                    print(point_iz_gb, point_iy_gb, point_ix_gb)
                    print(point_iz_local, point_iy_local, point_ix_local)   

print(keep_labels)

if len(keep_labels)>0: labels_in = update_labels(keep_labels, labels_in, cc_labels)

filename = param.data_path+"/"+param.sample_name+"/"+"Zebrafinch-input_labels_discarded-"+str(bz).zfill(4)+"z-"+str(by).zfill(4)+"y-"+str(bx).zfill(4)+"x"

writeData(filename, labels_in)
