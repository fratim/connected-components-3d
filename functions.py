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

# set will be deprecated soon on numba, but until now an alternative has not been implemented
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def ReadH5File(filename,box):
    # return the first h5 dataset from this file
    with h5py.File(filename, 'r') as hf:
        keys = [key for key in hf.keys()]
        d = hf[keys[0]]
        # read entire chunk if first element of box is one and length is one
        if len(box)==1 and box[0]==1:
            data=np.array(d)
        else:
            data = np.array(d[box[0]:box[1],box[2]:box[3],box[4]:box[5]])
    return data

def WriteH5File(data, filename, dataset):
    with h5py.File(filename, 'w') as hf:
        # should cover all cases of affinities/images
        hf.create_dataset(dataset, data=data, compression='gzip')

#read data from HD5, given the file path
def readData(box, filename):
    # read in data block
    filename_comp = filename +".h5"
    data_in = ReadH5File(filename_comp, box)
    labels = data_in

    return labels

# write data to H5 file
def writeData(filename,labels):

    filename_comp = filename +".h5"
    WriteH5File(data=labels, filename=filename_comp, dataset="main")

#compute the connected Com ponent labels
def computeConnectedComp26(labels):
    connectivity = 26 # only 26, 18, and 6 are allowed
    cc_labels = cc3d.connected_components(labels, connectivity=connectivity, max_labels=45000000)

    # You can extract individual components like so:
    n_comp = np.max(cc_labels) + 1

    del cc_labels

    return n_comp

#compute the connected Com ponent labels
def computeConnectedComp6(labels, start_label, max_labels):
    connectivity = 6 # only 26, 18, and 6 are allowed
    cc_labels = cc3d.connected_components(labels, connectivity=connectivity, max_labels=max_labels)

    n_comp = (np.min(cc_labels)*-1)

    if start_label!=-1:
        cc_labels[cc_labels<0] = cc_labels[cc_labels<0] + start_label

    return cc_labels, n_comp

# connect 2 wall parts
def conntectWalls(label_set, output_path, bz, by, bx, axis):

    if axis == "z":
        bz_min = bz+1
        by_min = by
        bx_min = bx
    elif axis == "y":
        bz_min = bz
        by_min = by+1
        bx_min = bx
    elif axis == "x":
        bz_min = bz
        by_min = by
        bx_min = bx+1
    else:
        raise ValueError("Unknown axis!")

    print("making forward connection in " + axis, flush=True)

    # load wall of current block
    output_folder_max = blockFolderPath(output_path,bz,by,bx)
    MaxWall = readData(box=[1], filename=output_folder_max+axis+"MaxWall")

    # load neighboring wall in positive z direction
    output_folder_min = blockFolderPath(output_path,bz_min,by_min,bx_min)
    MinWall = readData(box=[1], filename=output_folder_min+axis+"MinWall")

    # check dimensions
    if MinWall.shape != MaxWall.shape: raise ValueError("Walls dont have same dimension!!")

    label_set = noPythonConnectWalls(label_set, MinWall, MaxWall)

    return label_set

def conntectWalltoBorder(label_set, output_path, bz, by, bx, axis, direction):

    if direction != "Max" and direction != "Min":
        raise ValueError("Direction must be Max or Min")

    print("making "  + direction + " connection in " + axis, flush=True)

    output_folder_max = blockFolderPath(output_path,bz,by,bx)
    wall = readData(box=[1], filename=output_folder_max+axis+direction+"Wall")

    label_set = noPythonWallBorder(label_set, wall)

    return label_set

@njit
def noPythonWallBorder(label_set, wall):
    for ia in range(wall.shape[0]):
        for ib in range(wall.shape[1]):
            label_set.add((wall[ia,ib],0x7FFFFFFFFFFFFFFF))
    return label_set

@njit
def noPythonConnectWalls(label_set, MinWall, MaxWall):
    # check dimensions
    if MinWall.shape != MaxWall.shape: raise ValueError("Walls dont have same dimension!!")

    for ia in range(MaxWall.shape[0]):
        for ib in range(MaxWall.shape[1]):
            label_set.add((MaxWall[ia,ib],MinWall[ia,ib]))
            label_set.add((MinWall[ia,ib],MaxWall[ia,ib]))

    return label_set

# find sets of adjacent components
def findAdjLabelSetGlobal(output_path, label_set, yres, xres, border_contact,bz,by,bx):

    # Z direction
    ###################
    if border_contact[0]==1:
        # connect min z wall to border
        label_set = conntectWalltoBorder(label_set=label_set, output_path=output_path, bz=bz, by=by, bx=bx, axis="z", direction="Min")
    #check if no border contact in z direction
    if border_contact[1]==0:
        # connect max z wall to next blocks z min wall
        label_set = conntectWalls(label_set=label_set, output_path=output_path, bz=bz, by=by, bx=bx, axis="z")
    elif border_contact[1]==1:
        # connect max z wall to border
        label_set = conntectWalltoBorder(label_set=label_set, output_path=output_path, bz=bz, by=by, bx=bx, axis="z", direction="Max")
    else:
        raise ValueError("Unknown Error")

    # Y direction
    ###################
    if border_contact[2]==1:
        # connect min z wall to border
        label_set = conntectWalltoBorder(label_set=label_set, output_path=output_path, bz=bz, by=by, bx=bx, axis="y", direction="Min")
    #check if no border contact in z direction
    if border_contact[3]==0:
        # connect max z wall to next blocks z min wall
        label_set = conntectWalls(label_set=label_set, output_path=output_path, bz=bz, by=by, bx=bx, axis="y")
    elif border_contact[3]==1:
        # connect max z wall to border
        label_set = conntectWalltoBorder(label_set=label_set, output_path=output_path, bz=bz, by=by, bx=bx, axis="y", direction="Max")
    else:
        raise ValueError("Unknown Error")


    # X direction
    ###################
    if border_contact[4]==1:
        # connect min z wall to border
        label_set = conntectWalltoBorder(label_set=label_set, output_path=output_path, bz=bz, by=by, bx=bx, axis="x", direction="Min")
    #check if no border contact in z direction
    if border_contact[5]==0:
        # connect max z wall to next blocks z min wall
        label_set = conntectWalls(label_set=label_set, output_path=output_path, bz=bz, by=by, bx=bx, axis="x")
    elif border_contact[5]==1:
        # connect max z wall to border
        label_set = conntectWalltoBorder(label_set=label_set, output_path=output_path, bz=bz, by=by, bx=bx, axis="x", direction="Max")
    else:
        raise ValueError("Unknown Error")

    return label_set

# find sets of adjacent components
@njit
def findAdjLabelSetLocal(cc_labels, yres, xres):

    neighbor_label_set_inside = set()
    neighbor_label_set_border = set()

    box = [0,cc_labels.shape[0],0,cc_labels.shape[0],0,cc_labels.shape[0]]

    for iz in range(0, box[1]-box[0]-1):
        for iy in range(0, box[3]-box[2]-1):
            for ix in range(0, box[5]-box[4]-1):

                curr_comp = cc_labels[iz,iy,ix]

                if curr_comp != cc_labels[iz+1,iy,ix]:
                    neighbor_label_set_inside.add((cc_labels[iz,iy,ix],cc_labels[iz+1,iy,ix]))
                    neighbor_label_set_inside.add((cc_labels[iz+1,iy,ix],cc_labels[iz,iy,ix]))

                if curr_comp != cc_labels[iz,iy+1,ix]:
                    neighbor_label_set_inside.add((cc_labels[iz,iy,ix],cc_labels[iz,iy+1,ix]))
                    neighbor_label_set_inside.add((cc_labels[iz,iy+1,ix],cc_labels[iz,iy,ix]))

                if curr_comp != cc_labels[iz,iy,ix+1]:
                    neighbor_label_set_inside.add((cc_labels[iz,iy,ix],cc_labels[iz,iy,ix+1]))
                    neighbor_label_set_inside.add((cc_labels[iz,iy,ix+1],cc_labels[iz,iy,ix]))

    for iz in [0, box[1]-box[0]-1]:
        for iy in range(0, box[3]-box[2]):
            for ix in range(0, box[5]-box[4]):

                #interconnect in plane
                curr_comp = cc_labels[iz,iy,ix]

                if (iy+1) < box[3]-box[2]:
                    if curr_comp != cc_labels[iz,iy+1,ix]:
                        neighbor_label_set_inside.add((cc_labels[iz,iy,ix],cc_labels[iz,iy+1,ix]))
                        neighbor_label_set_inside.add((cc_labels[iz,iy+1,ix],cc_labels[iz,iy,ix]))

                if (ix+1) < box[5]-box[4]:
                    if curr_comp != cc_labels[iz,iy,ix+1]:
                        neighbor_label_set_inside.add((cc_labels[iz,iy,ix],cc_labels[iz,iy,ix+1]))
                        neighbor_label_set_inside.add((cc_labels[iz,iy,ix+1],cc_labels[iz,iy,ix]))

                # write dict of border components
                neighbor_label_set_border.add((cc_labels[iz,iy,ix], 0x7FFFFFFFFFFFFFFF))

    for iz in range(0, box[1]-box[0]):
        for iy in [0, box[3]-box[2]-1]:
            for ix in range(0, box[5]-box[4]):

                #interconnect in plane
                curr_comp = cc_labels[iz,iy,ix]

                if (iz+1) < box[1]-box[0]:
                    if curr_comp != cc_labels[iz+1,iy,ix]:
                        neighbor_label_set_inside.add((cc_labels[iz,iy,ix],cc_labels[iz+1,iy,ix]))
                        neighbor_label_set_inside.add((cc_labels[iz+1,iy,ix],cc_labels[iz,iy,ix]))

                if (ix+1) < box[5]-box[4]:
                    if curr_comp != cc_labels[iz,iy,ix+1]:
                        neighbor_label_set_inside.add((cc_labels[iz,iy,ix],cc_labels[iz,iy,ix+1]))
                        neighbor_label_set_inside.add((cc_labels[iz,iy,ix+1],cc_labels[iz,iy,ix]))

                neighbor_label_set_border.add((cc_labels[iz,iy,ix], 0x7FFFFFFFFFFFFFFF))

    for iz in range(0, box[1]-box[0]):
        for iy in range(0, box[3]-box[2]):
            for ix in [0, box[5]-box[4]-1]:

                #interconnect in plane
                curr_comp = cc_labels[iz,iy,ix]

                if (iz+1) < box[1]-box[0]:
                    if curr_comp != cc_labels[iz+1,iy,ix]:
                        neighbor_label_set_inside.add((cc_labels[iz,iy,ix],cc_labels[iz+1,iy,ix]))
                        neighbor_label_set_inside.add((cc_labels[iz+1,iy,ix],cc_labels[iz,iy,ix]))

                if (iy+1) < box[3]-box[2]:
                    if curr_comp != cc_labels[iz,iy+1,ix]:
                        neighbor_label_set_inside.add((cc_labels[iz,iy,ix],cc_labels[iz,iy+1,ix]))
                        neighbor_label_set_inside.add((cc_labels[iz,iy+1,ix],cc_labels[iz,iy,ix]))

                neighbor_label_set_border.add((cc_labels[iz,iy,ix], 0x7FFFFFFFFFFFFFFF))

    return neighbor_label_set_inside, neighbor_label_set_border

# set dict value undetermined set to 0
def setUndeterminedtoNonHole(undetermined, associated_label):

    for _ in range(len(undetermined)):

        elem = undetermined.pop()
        associated_label[elem] = 0

    if len(undetermined)>0:
        raise ValueError("Uknown Error")

    return associated_label

# get neighbor label dict from neighbor label set
def writeNeighborLabelDict(neighbor_label_dict, neighbor_label_set):

    if neighbor_label_dict == False:
        neighbor_label_dict = dict()

    for s in range(len(neighbor_label_set)):
        pair = neighbor_label_set.pop()
        if pair[0]<0:
            if pair[0] in neighbor_label_dict.keys():
                if pair[1] not in neighbor_label_dict[pair[0]]:
                    neighbor_label_dict[pair[0]].append(pair[1])
                else:
                    continue
            else:
                neighbor_label_dict[pair[0]] = [pair[1]]
        else:
            continue

    return neighbor_label_dict

# create string of connected components that are a whole
def findAssociatedLabels(neighbor_label_dict, undetermined, associated_label):

    # time_start = time.time()
    border_contact = set()
    isHole = set()
    isNotHole = set()

    counter = 0

    while len(undetermined)>0:

        counter = counter + 1
        if counter%1000==0:
            print("Counter is: " + str(counter), flush=True)
        query_comp = undetermined.pop()

        #check if it has only one neighbor and this neighbor is a neuron
        if len(neighbor_label_dict[query_comp])==1 and neighbor_label_dict[query_comp][0]!=0x7FFFFFFFFFFFFFFF and neighbor_label_dict[query_comp][0]>0:
                associated_label[query_comp] = neighbor_label_dict[query_comp][0]
                isHole.add(query_comp)

        # otherwise unroll neighbors to identify
        else:

            # list of nodes to expand
            open = []
            # iterate over all neighbors and add them to the open set, if they are a background componente (i.e. are negative)
            for elem in neighbor_label_dict[query_comp]:
                if elem == 0x7FFFFFFFFFFFFFFF:
                    continue
                elif elem < 0:
                    for son in neighbor_label_dict[elem]:
                        if son not in neighbor_label_dict[query_comp]:
                            neighbor_label_dict[query_comp].append(son)
                            if son<0:
                                open.insert(0,son)
            # appen all negative background components that are neighbors or ancestors
            while len(open)>0:
                elem = open.pop()
                if elem == 0x7FFFFFFFFFFFFFFF:
                    if 0x7FFFFFFFFFFFFFFF not in neighbor_label_dict[query_comp]:
                        neighbor_label_dict[query_comp].append(0x7FFFFFFFFFFFFFFF)
                else:
                    for son in neighbor_label_dict[elem]:
                        if son not in neighbor_label_dict[query_comp]:
                            neighbor_label_dict[query_comp].append(son)
                            if son<0:
                                open.insert(0,son)

            # check if there is a bordercontact, then add to bordercontact but remove all elemnts from undetermined (will be added again later)
            if 0x7FFFFFFFFFFFFFFF in neighbor_label_dict[query_comp]:
                border_contact.add(query_comp)
                for elem in neighbor_label_dict[query_comp]:
                    if elem < 0:
                        border_contact.add(elem)
                        undetermined.discard(elem)

            # if component does not have border contact, it can now be definitley determined if it is a hole or not
            else:
            # check again if there is only one positive neighbor and that it is not boundary and it is a neuron, if so, it is a hole
                if len(list(filter(lambda a: a>0, neighbor_label_dict[query_comp])))==1:
                    associated_label[query_comp] = np.max(neighbor_label_dict[query_comp])
                    isHole.add(query_comp)
                    for elem in neighbor_label_dict[query_comp]:
                        if elem < 0:
                            associated_label[elem]=np.max(neighbor_label_dict[query_comp])
                            undetermined.discard(elem)
                            isHole.add(elem)
                else:
                    associated_label[query_comp] = 0
                    for elem in neighbor_label_dict[query_comp]:
                        if elem < 0:
                            associated_label[elem]=0
                            undetermined.discard(elem)
                            isNotHole.add(query_comp)
            # delte open set
            del open

    if len(undetermined)>0:
        raise ValueError("Unknown Error")

    undetermined = undetermined.union(border_contact)

    return associated_label, undetermined, isHole, isNotHole

def removeDetComp(neighbor_label_set, isHole, isNotHole):

    neighbor_label_set_out = set()

    for _ in range(len(neighbor_label_set)):
        elem = neighbor_label_set.pop()
        if elem[0] in isHole or elem[0] in isNotHole:
            continue
            print("removed: " + str(elem), flush=True)
        else:
            neighbor_label_set_out.add(elem)

    return neighbor_label_set_out

# fill detedted wholes and give non_wholes their ID (for visualization)
def fillWholes(output_path,associated_label, bz):

    # create filename
    input_name = "cc_labels"
    box = [1]

    # read in data
    cc_labels = readData(box, output_path+input_name)

    box = [0,cc_labels.shape[0],0,cc_labels.shape[1],0,cc_labels.shape[2]]

    # use nopython to do actual computation
    cc_labels = fillwholesNoPython(box,cc_labels,associated_label)
    output_name = str(bz*128).zfill(4)
    writeData(output_path+output_name, cc_labels)

@njit
def fillwholesNoPython(box,cc_labels,associated_label):
    for iz in range(box[0], box[1]):
        for iy in range(box[2], box[3]):
            for ix in range(box[4], box[5]):

                if cc_labels[iz,iy,ix] < 0:
                    cc_labels[iz,iy,ix] = associated_label[cc_labels[iz,iy,ix]]
                else:
                    continue

    return cc_labels

def concatFiles(box, slices_s, slices_e, output_path, data_path):

    for i in range(slices_s,slices_e+1):
        sample_name = str(i*128).zfill(4)
        if i is slices_s:
            labels_concat = readData(box, data_path+sample_name)
        else:
            labels_temp = readData(box, data_path+sample_name)
            labels_old = labels_concat.copy()
            del labels_concat
            labels_concat = np.concatenate((labels_old,labels_temp),axis=0)
            del labels_temp

    print("Concat size/ shape: " + str(labels_concat.nbytes) + '/ ' + str(labels_concat.shape), flush=True)
    writeData(output_path, labels_concat)

    del labels_concat

def concatBlocks(z_start, y_start, x_start, n_blocks_z, n_blocks_y, n_blocks_x, bs_z, bs_y, bs_x, output_path):

    for bz in range(z_start, z_start+n_blocks_z):
        print("processing z block " + str(bz), flush=True)
        for by in range(y_start, y_start+n_blocks_y):
            for bx in range(x_start, x_start+n_blocks_x):


                input_name = "/z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)+"/" + str(bz*128).zfill(4)

                if bz==z_start and by==y_start and bx==x_start:
                    labels_concat =  np.zeros((bs_z*n_blocks_z,bs_y*n_blocks_y,bs_x*n_blocks_x),dtype=np.uint16)
                box=[0,bs_z,0,bs_y,0,bs_x]
                labels_concat[(bz-z_start)*bs_z:((bz-z_start)+1)*bs_z,(by-y_start)*bs_y:(by-y_start+1)*bs_y,(bx-x_start)*bs_x:(bx-x_start+1)*bs_x] = readData(box, filename=output_path+input_name)
    print("Concat size/ shape: " + str(labels_concat.nbytes) + '/ ' + str(labels_concat.shape), flush=True)
    output_name = "filled"
    writeData(output_path+output_name, labels_concat)

    return labels_concat

@njit
def IdxToIdi(iv, yres, xres):
    iz = iv // (yres * xres)
    iy = (iv - iz * yres * xres) // xres
    ix = iv % xres
    return iz, iy, ix

@njit
def IdiToIdx(iz, iy, ix, yres, xres):
    if (iz<param.bs_z*param.z_start or iy<param.bs_y*param.y_start or ix<param.bs_x*param.x_start
            or iy >= param.bs_y*(param.y_start+param.n_blocks_y) or ix >= param.bs_x*(param.x_start+param.n_blocks_x)):
        raise ValueError("Out of bounds IditoIdx")
        return -1
    else:
        return iz * yres * xres + iy * xres + ix

def dumpNumbaDictToFile(object, object_name, output_path, output_name):
    filename = output_path+object_name+output_name+".pickle"
    temp = dict()
    temp.update(object)

    f = open(filename, 'wb')
    pickle.dump(temp, f)
    f.close()

def dumpToFile(object, object_name, output_path, output_name):
    filename = output_path+object_name+output_name+".pickle"
    f = open(filename, 'wb')
    pickle.dump(object, f)
    f.close()

def readFromFile(object_name, output_path, output_name):
    filename = output_path+object_name+output_name+".pickle"
    f = open(filename, 'rb')
    object = pickle.load(f)
    f.close()

    return object

class dataBlock:

    def __init__(self,viz_wholes):
        self.vizWholes = viz_wholes

    def useExistingFolder(self, output_path, sample_name):
        self.folder_path = output_path + sample_name + "/"
        self.sample_name = sample_name
        self.bs_y = int(sample_name[-9:-5])
        self.bs_x = int(sample_name[-4:])

    def concatFiles(self):

        for i in range(self.slices_start,self.slices_end+1):
            sample_name = str(i*128).zfill(4)
            if i is self.slices_start:
                labels_concat = readData(self.box_concat, self.data_path+sample_name)
            else:
                labels_temp = readData(self.box_concat, self.data_path+sample_name)
                labels_old = labels_concat.copy()
                del labels_concat
                labels_concat = np.concatenate((labels_old,labels_temp),axis=0)
                del labels_temp

        print("Concat size/ shape: " + str(labels_concat.nbytes) + '/ ' + str(labels_concat.shape), flush=True)
        writeData(self.folder_path+self.sample_name, labels_concat)

        del labels_concat

    def evaluateWholes(self, ID_A, ID_B):
        print("Evaluating wholes...", flush=True)
        # load gt wholes
        gt_wholes_filepath = self.folder_path+"/"+ID_A+"/"+"wholes"
        box = [1]
        wholes_gt = readData(box, gt_wholes_filepath)

        # load block wholes
        inBlocks_wholes_filepath = self.folder_path+"/"+ID_B+"/"+"wholes"
        box = [1]
        wholes_inBlocks = readData(box, inBlocks_wholes_filepath)

        try:# check that both can be converted to int16
            if np.max(wholes_gt)>32767 or np.max(wholes_inBlocks)>32767:
                raise ValueError("Cannot convert wholes to int16 (max is >32767)")
        except:
            print("Cannot convert wholes to int16 (max is >32767) -  ignored this Error", flush=True)

        wholes_gt = wholes_gt.astype(np.int16)
        wholes_inBlocks = wholes_inBlocks.astype(np.int16)
        wholes_gt = np.subtract(wholes_gt, wholes_inBlocks)
        diff = wholes_gt
        # free some RAM
        del wholes_gt, wholes_inBlocks

        print("Freed memory", flush=True)

        if np.min(diff)<0:
            FP = diff.copy()
            FP[FP>0]=0
            n_points_FP = np.count_nonzero(FP)
            n_comp_FP = computeConnectedComp26(FP)-1
            print("FP classifications (points/components): " + str(n_points_FP) + "/ " +str(n_comp_FP), flush=True)

            # unique_values = np.unique(FP)
            # for u in unique_values:
            #     if u!=0:
            #         print("Coordinates of component " + str(u))
            #         coods = np.argwhere(FP==u)
            #         for i in range(coods.shape[0]):
            #             print(str(coods[i,0]) + ", " + str(coods[i,1]) + ", " + str(coods[i,2]))

            del FP
        else:
            print("No FP classification", flush=True)

        if np.max(diff)>0:
            FN = diff.copy()
            FN[FN<0]=0
            n_points_FN = np.count_nonzero(FN)
            n_comp_FN = computeConnectedComp26(FN)-1
            print("FN classifications (points/components): " + str(n_points_FN) + "/ " +str(n_comp_FN), flush=True)
            del FN

        else:
            print("No FN classification", flush=True)

        output_name = 'diff_wholes_'+ID_A+"_"+ID_B
        writeData(self.folder_path+"/"+ID_B+"/"+output_name, diff)

        del diff

    def readLabels(self, data_path, sample_name, bz, by, bx):
        if param.isCluster:
        	filename = data_path+"/"+sample_name+"/"+str(bz*128).zfill(4)
        else:
        	filename = data_path+"/"+sample_name+"/"+"cut_z_"+ str((bz)).zfill(4)+"y_"+ str(by).zfill(4)+"x_"+ str(bx).zfill(4)
        box = [1]
        self.labels_in = readData(box, filename)
        self.bs_z = self.labels_in.shape[0]
        self.bs_y = self.labels_in.shape[1]
        self.bs_x = self.labels_in.shape[2]
        self.bz=bz
        self.by=by
        self.bx=bx

    def computeStepOne(self, label_start, output_path):

        start_time_cc_labels = time.time()
        # compute connected component labels
        cc_labels, n_comp = computeConnectedComp6(self.labels_in,label_start,param.max_labels_block)

        del self.labels_in

        output_name = "cc_labels"
        output_folder = blockFolderPath(output_path,self.bz,self.by,self.bx)

        #save output and slices of walls
        makeFolder(output_folder)
        writeData(output_folder+output_name, cc_labels)
        writeData(output_folder+"zMinWall", cc_labels[0 ,: ,: ])
        writeData(output_folder+"zMaxWall", cc_labels[-1,: ,: ])
        writeData(output_folder+"yMinWall", cc_labels[: ,0 ,: ])
        writeData(output_folder+"yMaxWall", cc_labels[: ,-1,: ])
        writeData(output_folder+"xMinWall", cc_labels[: ,: ,0 ])
        writeData(output_folder+"xMaxWall", cc_labels[: ,: ,-1])

        self.time_cc_labels = time.time()-start_time_cc_labels
        start_time_AdjLabelLocal = time.time()

        # find the set of adjacent labels, both inside the volume and the ones connected to the local border
        neighbor_label_set_inside_local, neighbor_label_set_border_local = findAdjLabelSetLocal(cc_labels, self.yres, self.xres)

        self.time_AdjLabelLocal = time.time()-start_time_AdjLabelLocal

        del cc_labels

        start_time_assoc_labels = time.time()

        # for identification of local wholes that do not cross the border, unify both sets and write a dict of the corresponding neighbors for each component
        neighbor_label_set = neighbor_label_set_inside_local.union(neighbor_label_set_border_local)
        neighbor_label_dict = writeNeighborLabelDict(neighbor_label_dict=False, neighbor_label_set=neighbor_label_set)

        # create a set of undtermined components, at this stage all components in the block and find associated labels of components that can be identified already
        undetermined_local = set(neighbor_label_dict.keys())
        associated_label_local = Dict.empty(key_type=types.int64,value_type=types.int64)
        associated_label_local, undetermined_local, isHole, isNotHole = findAssociatedLabels(neighbor_label_dict=neighbor_label_dict, undetermined=undetermined_local, associated_label=associated_label_local)

        self.time_assoc_labels = time.time()-start_time_assoc_labels

        del neighbor_label_set, neighbor_label_set_border_local, neighbor_label_dict

        self.n_comp = n_comp
        self.n_Holes = len(isHole)
        self.n_NotHoles = len(isNotHole)

        # remove alrady detected hole components from neighbor label set local and write the according neighbor label dict of components that are not yet determined
        neighbor_label_set_inside_local_reduced  = removeDetComp(neighbor_label_set_inside_local.copy(), isHole, isNotHole)
        neighbor_label_dict_reduced = writeNeighborLabelDict(neighbor_label_dict=False, neighbor_label_set=neighbor_label_set_inside_local_reduced)
        self.size_label_set_inside = len(neighbor_label_set_inside_local)
        self.size_label_set_inside_reduced = len(neighbor_label_set_inside_local_reduced)

        # write components that are needed for later steps to files
        output_folder = blockFolderPath(output_path,self.bz,self.by,self.bx)
        start_time_writepickle = time.time()
        dumpNumbaDictToFile(associated_label_local, "associated_label_local", output_folder, "")
        dumpToFile(undetermined_local, "undetermined_local", output_folder, "")
        dumpToFile(neighbor_label_dict_reduced, "neighbor_label_dict_reduced", output_folder, "")

        self.time_writepickle = time.time()-start_time_writepickle

    def setRes(self, zres,yres,xres):
        self.zres = zres
        self.yres = yres
        self.xres = xres

def compareOutp(output_path, sample_name, ID_B):
    vizWholes = True

    blockA = dataBlock(viz_wholes=vizWholes)

    blockA.useExistingFolder(output_path=output_path, sample_name=sample_name)

    blockA.evaluateWholes(ID_A="gt", ID_B=ID_B)

def makeFolder(folder_path):
    if os.path.exists(folder_path):
        raise ValueError("Folderpath " + folder_path + " already exists!")
    else:
        os.mkdir(folder_path)

def blockFolderPath(output_path,bz,by,bx):
    return output_path+"/z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)+"/"

if __name__== "__main__":
  main()
