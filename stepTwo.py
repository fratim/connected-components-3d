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


from functions import readFromFile, findAdjLabelSetGlobal, writeNeighborLabelDict, findAssociatedLabels, setUndeterminedtoNonHole, dumpNumbaDictToFile, makeFolder, dumpToFile, IdiToIdx, combineBoxes

print("executing Step 2 calculations...", flush=True)

# STEP 2

# Iteration 1 interconnects 2 z-blocks, also interconnecting in all x-y directions and loading the needed variables

neighbor_label_set_inside_global = set()
associated_label_global = Dict.empty(key_type=types.int64,value_type=types.int64)
undetermined_global = set()
neighbor_label_set_border_global = {(1,1)}

z_range = np.arange(param.z_start, param.z_start+param.n_blocks_z)

#iteration 1 (2 blokcs)
for bz_global in z_range[::2]:

    border_comp_combined = Dict.empty(key_type=types.int64,value_type=types.int64)
    border_comp_exist_combined = set()

    # determine if the last block is single (uneven number of blocks)
    if bz_global+1>z_range[-1]:
        bz_range = [bz_global]
        isSingle = True
        print("ISSINGLE!")
    elif bz_global+1<=z_range[-1]:
        bz_range = [bz_global, bz_global+1]
        isSingle = False
    else:
        raise ValueError("Unknown Error!")

    for bz in bz_range:
        for by in range(param.y_start, param.y_start+param.n_blocks_y):
            for bx in range(param.x_start, param.x_start+param.n_blocks_x):

                print("Block z is: " + str(bz), flush=True)
                output_folder = param.folder_path+"/z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)+"/"

                border_comp_local = readFromFile("border_comp_local", output_folder, "")
                border_comp_exist_local = readFromFile("border_comp_exist_local", output_folder, "")
                neighbor_label_set_inside_local = readFromFile("neighbor_label_set_inside_local", output_folder, "")
                associated_label_local = readFromFile("associated_label_local", output_folder, "")
                undetermined_local = readFromFile("undetermined_local", output_folder, "")

                border_comp_combined.update(border_comp_local)
                border_comp_exist_combined = border_comp_exist_combined.union(border_comp_exist_local)

                neighbor_label_set_inside_global = neighbor_label_set_inside_global.union(neighbor_label_set_inside_local)
                associated_label_global.update(associated_label_local)
                undetermined_global = undetermined_global.union(undetermined_local)

                del border_comp_local, border_comp_exist_local, neighbor_label_set_inside_local, associated_label_local, undetermined_local

    border_comp_combined_new = Dict.empty(key_type=types.int64,value_type=types.int64)
    border_comp_exist_combined_new = {(2**30)}



    for bz in bz_range:

        if isSingle:
            connectInPosZdirec = False
            connectInNegZdirec = False

        elif bz == bz_range[0]:
            connectInPosZdirec = True
            connectInNegZdirec = False

        elif bz == bz_range[1]:
            connectInPosZdirec = False
            connectInNegZdirec = True

        else:
            raise ValueError("Unkown Error")

        for by in range(param.y_start, param.y_start+param.n_blocks_y):
            for bx in range(param.x_start, param.x_start+param.n_blocks_x):

                box = [bz*param.bs_z,(bz+1)*param.bs_z,by*param.bs_y,(by+1)*param.bs_y,bx*param.bs_x,(bx+1)*param.bs_x]


                border_comp_combined_new, border_comp_exist_combined_new, neighbor_label_set_border_global = findAdjLabelSetGlobal(box, neighbor_label_set_border_global,
                                                        border_comp_combined, border_comp_exist_combined, param.yres, param.xres, connectInPosZdirec, connectInNegZdirec,
                                                        border_comp_combined_new, border_comp_exist_combined_new)

    border_comp_exist_combined_new.remove((2**30))

    if isSingle:
        box_combined = [bz_global*param.bs_z,(bz_global+1)*param.bs_z,param.y_start*param.bs_y,(param.y_start+param.n_blocks_y)*param.bs_y,
                            param.x_start*param.bs_x,(param.x_start+param.n_blocks_x)*param.bs_x]
    elif not isSingle:
        box_combined = [bz_global*param.bs_z,(bz_global+2)*param.bs_z,param.y_start*param.bs_y,(param.y_start+param.n_blocks_y)*param.bs_y,
                            param.x_start*param.bs_x,(param.x_start+param.n_blocks_x)*param.bs_x]

    print("Box combined: " + str(box_combined))

    iteration = 1
    output_folder = param.folder_path+"/z"+str(bz_global).zfill(4)+"_it_"+str(iteration)+"/"
    makeFolder(output_folder)
    dumpNumbaDictToFile(border_comp_combined_new, "border_comp_local", output_folder, "")
    dumpToFile(border_comp_exist_combined_new, "border_comp_exist_local", output_folder, "")
    dumpToFile(box_combined, "box_combined", output_folder, "")

    del border_comp_combined, border_comp_exist_combined, border_comp_combined_new, border_comp_exist_combined_new, box_combined

iteration = 2
lastIteration = False
while lastIteration==False:

    print("------------------------------------------------")
    print("Iteration " + str(iteration))
    block_size = 2**iteration
    bz_global_range = z_range[::block_size]

    if z_range[0]+block_size > z_range[-1]:
        print("range: " + str(z_range))
        print("block size: " + str(block_size))
        lastIteration = True
        print ("this is last iteration")


    #iteration2 (4 blocks)
    bz_global_range = z_range[::block_size]
    for bz_global in bz_global_range:

        border_comp_combined = Dict.empty(key_type=types.int64,value_type=types.int64)
        border_comp_exist_combined = set()

        # check if block is missing and have to compute single
        if bz_global+int(block_size/2)>z_range[-1]:
            bz_range = [bz_global]
            isSingle = True
            print("ISSINGLE!")
        elif bz_global+int(block_size/2)<=z_range[-1]:
            bz_range = [bz_global, bz_global+int(block_size/2)]
            isSingle = False
        else:
            raise ValueError("Unknown Error!")

        for bz in bz_range:

            print("bz read in is : " + str(bz), flush=True)
            output_folder = param.folder_path+"/z"+str(bz).zfill(4)+"_it_"+str(iteration-1)+"/"

            if bz==bz_range[0]: box_a = readFromFile("box_combined", output_folder, "")
            if bz==bz_global+int(block_size/2): box_b = readFromFile("box_combined", output_folder, "")

            border_comp_local = readFromFile("border_comp_local", output_folder, "")
            border_comp_exist_local = readFromFile("border_comp_exist_local", output_folder, "")

            border_comp_combined.update(border_comp_local)
            border_comp_exist_combined = border_comp_exist_combined.union(border_comp_exist_local)

            del border_comp_local, border_comp_exist_local

        if isSingle:
            box_combined = box_a
        if not isSingle:
            box_combined = combineBoxes(box_a, box_b)

        border_comp_combined_new = Dict.empty(key_type=types.int64,value_type=types.int64)
        border_comp_exist_combined_new = {(2**30)}

        for bz in bz_range:

            if lastIteration:
                connectInPosZdirec = True
                connectInNegZdirec = True
            elif isSingle:
                connectInPosZdirec = False
                connectInNegZdirec = False
            else:
                if bz == bz_range[0]:
                    connectInPosZdirec = True
                    connectInNegZdirec = False
                elif bz == bz_range[1]:
                    connectInPosZdirec = False
                    connectInNegZdirec = True
                else:
                    raise ValueError("Error in determining z connection directions")

            print("Z Pos: " + str(connectInPosZdirec))
            print("Z Neg: " + str(connectInNegZdirec))

            if bz == bz_range[0]:
                box = box_a
            elif not isSingle:
                if bz == bz_range[1]:
                    box = box_b
                else:
                    raise ValueError("Unknown Error")
            else:
                raise ValueError("Unknown Error")

            print(box)

            border_comp_combined_new, border_comp_exist_combined_new, neighbor_label_set_border_global = findAdjLabelSetGlobal(box, neighbor_label_set_border_global,
                                                    border_comp_combined, border_comp_exist_combined, param.yres, param.xres, connectInPosZdirec, connectInNegZdirec,
                                                    border_comp_combined_new, border_comp_exist_combined_new)

        print("bz_global write is : " + str(bz_global), flush=True)
        border_comp_exist_combined_new.remove((2**30))
        print("making folder: " + str(bz_global))
        output_folder = param.folder_path+"/z"+str(bz_global).zfill(4)+"_it_"+str(iteration)+"/"
        makeFolder(output_folder)
        dumpNumbaDictToFile(border_comp_combined_new, "border_comp_local", output_folder, "")
        dumpToFile(border_comp_exist_combined_new, "border_comp_exist_local", output_folder, "")
        dumpToFile(box_combined, "box_combined", output_folder, "")

        del border_comp_combined, border_comp_exist_combined, border_comp_combined_new, border_comp_exist_combined_new, box_combined

    iteration = iteration + 1

print("Created border_comp_exist_global and neighbor_label_set_border_global", flush=True)

neighbor_label_set_border_global.remove((1,1))
neighbor_label_set = neighbor_label_set_inside_global.union(neighbor_label_set_border_global)

print("Find associated labels...", flush=True)
neighbor_label_dict = writeNeighborLabelDict(neighbor_label_set)
associated_label_global, undetermined_global = findAssociatedLabels(neighbor_label_dict, undetermined_global, associated_label_global)
associated_label_global = setUndeterminedtoNonHole(undetermined_global, associated_label_global)

output_name = ""
dumpNumbaDictToFile(associated_label_global, "associated_label_global", param.folder_path, output_name)
