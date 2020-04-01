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
import math

from functions import makeFolder, dataBlock, readFromFile, findAdjLabelSetGlobal, dumpToFile, blockFolderPath, writeNeighborLabelDict, findAssociatedLabels, setUndeterminedtoNonHole, dumpNumbaDictToFile, fillWholes, readData


# set will be deprecated soon on numba, but until now an alternative has not been implemented
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

def execStep1(inplist):

    bz = inplist[0]
    by = inplist[1]

    print("STEP 1: running for bz,by: " + str(bz) + ", " + str(by))

    for bx in range(param.x_start, param.x_start + param.n_blocks_x):

        # compute and save variables and data
        start_time_read_labels = time.time()
        start_time_total = time.time()

        block_number = (bz)*(param.y_start+param.n_blocks_y)*(param.x_start+param.n_blocks_x)+by*(param.x_start+param.n_blocks_x)+bx
        label_start = -1*block_number*param.max_labels_block

        currBlock = dataBlock(viz_wholes=True)
        currBlock.readLabels(data_path=param.data_path,
                                bz=bz, by=by, bx=bx)

        time_read_labels = time.time()-start_time_read_labels

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

        # write total time to file
        g = open(param.total_time_filepath+"-"+str(bz).zfill(4)+"z-"+str(by).zfill(4)+"y-"+str(bx).zfill(4)+"x.txt", "w+")
        g.write(format(time_total, '.4f') + "\n")
        g.close()

        if param.compute_statistics:
            g = open(param.points_per_component_filepath+"-"+str(bz).zfill(4)+"z-"+str(by).zfill(4)+"y-"+str(bx).zfill(4)+"x.txt", "w+")
            for entry in currBlock.points_per_component.keys():
                if entry in currBlock.hole_components or entry in currBlock.undetermined:
                    g.write(str(int(entry)).zfill(25)+", "+str(int(currBlock.points_per_component[entry])).zfill(25)+"\n")
            g.close()

            g = open(param.hole_components_filepath+"-"+str(bz).zfill(4)+"z-"+str(by).zfill(4)+"y-"+str(bx).zfill(4)+"x.txt", "w+")
            for entry in currBlock.hole_components:
                g.write(str(int(entry)).zfill(25)+"\n")
            g.close()

    return 1

def execStep2A(inplist):

    bz = inplist[0]
    by = inplist[1]

    print("STEP 2A: running for bz,by: " + str(bz) + ", " + str(by))

    for bx in range(param.x_start, param.x_start + param.n_blocks_x):

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

        g = open(param.total_time_filepath+"-"+str(bz).zfill(4)+"z-"+str(by).zfill(4)+"y-"+str(bx).zfill(4)+"x.txt", "a+")
        g.write(format(time_total, '.4f') + "\n")
        g.close()

        if param.compute_statistics:
            g = open(param.component_equivalences_filepath+"-"+str(bz).zfill(4)+"z-"+str(by).zfill(4)+"y-"+str(bx).zfill(4)+"x.txt", "w+")
            for entry in neighbor_label_set_border_global:
                g.write(str(int(entry[0])).zfill(25)+", "+str(int(entry[1])).zfill(25)+"\n")
            g.close()

    return 1

def execStep2B(inpList):

    # input list is dummy argument

    print("STEP 2B: running")

    start_time_total = time.time()
    # final step
    neighbor_label_set_border_global_combined = set()
    associated_label_global = Dict.empty(key_type=types.int64,value_type=types.int64)
    undetermined_global = set()
    neighbor_label_dict = dict()

    #timing
    start_time_pickleload = time.time()

    # load sets by iterating over all folders (expected to be small time consumption)
    for bz in range(param.z_start, param.z_start+param.n_blocks_z):
        for by in range(param.y_start, param.y_start+param.n_blocks_y):
            for bx in range(param.x_start, param.x_start+param.n_blocks_x):

                output_folder = blockFolderPath(param.folder_path, bz,by,bx)

                neighbor_label_set_border_global = readFromFile("neighbor_label_set_border_global", output_folder, "")
                associated_label_local = readFromFile("associated_label_local", output_folder, "")
                undetermined_local = readFromFile("undetermined_local", output_folder, "")
                neighbor_label_dict_reduced_local = readFromFile("neighbor_label_dict_reduced", output_folder, "")

                neighbor_label_set_border_global_combined = neighbor_label_set_border_global_combined.union(neighbor_label_set_border_global)
                associated_label_global.update(associated_label_local)
                undetermined_global = undetermined_global.union(undetermined_local)
                neighbor_label_dict.update(neighbor_label_dict_reduced_local)

                del neighbor_label_set_border_global, associated_label_local, undetermined_local

    len_label_set_border = len(neighbor_label_set_border_global_combined)

    time_pickleload = time.time() - start_time_pickleload
    start_time_findAssocLabelGlobal = time.time()

    #compute associated label global
    neighbor_label_dict = writeNeighborLabelDict(neighbor_label_dict=neighbor_label_dict, neighbor_label_set=neighbor_label_set_border_global_combined.copy())
    associated_label_global, undetermined_global, isHole, isNotHole = findAssociatedLabels(neighbor_label_dict, undetermined_global, associated_label_global)
    associated_label_global = setUndeterminedtoNonHole(undetermined_global.copy(), associated_label_global)

    if param.compute_statistics:
        hole_components = isHole

    n_Holes = len(isHole)
    n_NotHoles = len(isNotHole)+len(undetermined_global)

    time_findAssocLabelGlobal = time.time() - start_time_findAssocLabelGlobal
    start_time_writepickle = time.time()

    dumpNumbaDictToFile(associated_label_global, "associated_label_global", param.folder_path, "")

    time_writepickle = time.time()-start_time_writepickle
    time_total = time.time()-start_time_total

    len_associated_label_global = len(associated_label_global)

    g = open(param.step02B_info_filepath, "a+")
    g.write(    "total_time," + format(time_total, '.4f') + "," +
                "pickleload_time," + format(time_pickleload, '.4f')+","+
                "findAssocLabelGlobal_time," + format(time_findAssocLabelGlobal, '.4f')+","+
                "writepickle_time," + format(time_writepickle, '.4f')+","+
                "len_label_set_border," + str(len_label_set_border).zfill(16)+","+
                "len_associated_label_global," + str(len_associated_label_global).zfill(16)+","+
                "n_Holes," + str(n_Holes).zfill(16)+","+
                "n_NotHoles," + str(n_NotHoles).zfill(16)+"\n")
    g.close()


    g = open(param.total_time_filepath+"-step_2B.txt", "a+")
    g.write(format(time_total, '.4f') + "\n")
    g.close()

    if param.compute_statistics:
        g = open(param.hole_components_filepath+"-global.txt", "a+")
        for entry in hole_components:
            g.write(str(int(entry)).zfill(25)+"\n")
        g.close()

    return 1

def execStep3(inplist):

    bz = inplist[0]
    by = inplist[1]

    print("STEP 3: running for bz,by: " + str(bz) + ", " + str(by))

    for bx in range(param.x_start, param.x_start + param.n_blocks_x):

        # STEP 3
        start_time_total = time.time()

        block_number = (bz)*(param.y_start+param.n_blocks_y)*(param.x_start+param.n_blocks_x)+by*(param.x_start+param.n_blocks_x)+bx
        label_start = -1*block_number*param.max_labels_block-1
        label_end = label_start - param.max_labels_block

        start_time_readpickle = time.time()
        associated_label_global = readFromFile("associated_label_global", param.folder_path, "")
        time_readpickle = time.time()-start_time_readpickle

        start_time_cutdict = time.time()
        associated_label_block = {key: value for key, value in associated_label_global.items() if (key>label_end and key<=label_start)}
        associated_label = Dict.empty(key_type=types.int64,value_type=types.int64)
        associated_label.update(associated_label_block)
        time_cutdict = time.time()-start_time_cutdict

        start_time_fillWholes = time.time()
        output_folder = param.folder_path+"/z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)+"/"
        fillWholes(output_path=output_folder,associated_label=associated_label, bz=bz, by=by, bx=bx)
        time_fillWholes = time.time() - start_time_fillWholes

        time_total = time.time()-start_time_total

        g = open(param.total_time_filepath+"-"+str(bz).zfill(4)+"z-"+str(by).zfill(4)+"y-"+str(bx).zfill(4)+"x.txt", "a+")
        g.write(format(time_total, '.4f') + "\n")
        g.close()

        g = open(param.step03_info_filepath, "a+")
        g.write(    "bz/by/bx,"+str(bz).zfill(4)+","+str(by).zfill(4)+","+str(bx).zfill(4)+","
                    "total_time," + format(time_total, '.4f') + "," +
                    "pickleload_time," + format(time_readpickle, '.4f')+","+
                    "cutdict_time," + format(time_cutdict, '.4f')+","+
                    "fillWholes_time," + format(time_fillWholes, '.4f')+"\n")
        g.close()

    return 1
