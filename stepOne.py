    import cc3d
    import numpy as np
    from dataIO import ReadH5File
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    import time
    from scipy.spatial import distance
    import h5py
    from numba import njit, types
    from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
    import warnings
    import scipy.ndimage.interpolation
    import math
    from numba.typed import Dict
    import os
    import psutil
    import sys
    import pickle

    # set will be deprecated soon on numba, but until now an alternative has not been implemented
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


    # STEP 1
    makeFolder(folder_path)

    #counters
    cell_counter = 0
    n_comp_total = 0

    for bz in range(z_start, z_start+n_blocks_z):
        print("processing z block " + str(bz))
        for by in range(y_start, y_start+n_blocks_y):
            for bx in range(x_start, x_start+n_blocks_x):

                block_number = (bz-6)*(y_start+n_blocks_y)*(x_start+n_blocks_x)+by*(x_start+n_blocks_x)+bx
                label_start = -1*block_number*max_labels_block -1

                currBlock = dataBlock(viz_wholes=True)
                currBlock.readLabels(data_path=data_path, sample_name=sample_name,
                                        bz=bz, by=by, bx=bx, bs_z=bs_z, bs_y=bs_y, bs_x=bs_x)
                currBlock.setRes(zres=zres,yres=yres,xres=xres)

                #TODO get rid off max labels block, just compute it in the connectedcomp function as the size of passed block
                currBlock.computeStepOne(label_start=label_start, max_labels_block=max_labels_block, output_path=folder_path)

                cell_counter += 1
                n_comp_total += currBlock.n_comp

                del currBlock

    print("n_comp_total: " + str(n_comp_total))
