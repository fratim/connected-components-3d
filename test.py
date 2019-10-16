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

#read data from HD5, given the file path
def readData(box, filename):
    # read in data block
    data_in = ReadH5File(filename, box)

    labels = data_in

    # print("data read in; shape: " + str(data_in.shape) + "; DataType: " + str(data_in.dtype))

    return labels

# get shape of data saved in H5
def getBoxAll(filename):

    # return the first h5 dataset from this file
    with h5py.File(filename, 'r') as hf:
        keys = [key for key in hf.keys()]
        d = hf[keys[0]]
        box = [0,d.shape[0],0,d.shape[1],0,d.shape[2]]

    return box

# write data to H5 file
def writeData(filename,labels):

    filename_comp = filename +".h5"

    with h5py.File(filename_comp, 'w') as hf:
        # should cover all cases of affinities/images
        hf.create_dataset("main", data=labels, compression='gzip')

#compute the connected Com ponent labels
def computeConnectedComp26(labels):
    connectivity = 26 # only 26, 18, and 6 are allowed
    cc_labels = cc3d.connected_components(labels, connectivity=connectivity, max_labels=45000000)

    # You can extract individual components like so:
    n_comp = np.max(cc_labels) + 1

    del cc_labels
    # print("Conntected Regions found: " + str(n_comp))

    # determine indices, numbers and counts for the connected regions
    # unique, counts = np.unique(cc_labels, return_counts=True)
    # print("Conntected regions and associated points: ")
    # print(dict(zip(unique, counts)))

    return n_comp

#compute the connected Com ponent labels
def computeConnectedComp6(labels, start_label, max_labels):
    connectivity = 6 # only 26, 18, and 6 are allowed
    cc_labels = cc3d.connected_components(labels, connectivity=connectivity, max_labels=max_labels)

    n_comp = (np.min(cc_labels)*-1)

    if start_label!=-1:
        cc_labels[cc_labels<0] = cc_labels[cc_labels<0] + start_label

    return cc_labels, n_comp

# find sets of adjacent components
@njit
def findAdjLabelSetGlobal(box, neighbor_label_set_border, border_comp, border_comp_exist, yres, xres):


    for iz in [0, box[1]-box[0]-1]:
        for iy in range(0, box[3]-box[2]):
            for ix in range(0, box[5]-box[4]):

                # connect to adjacent blocks
                if iz == 0 and IdiToIdx(iz+box[0]-1,iy+box[2],ix+box[4],yres,xres) in border_comp_exist:
                    neighbor_label_set_border.add((border_comp[IdiToIdx(iz+box[0],iy+box[2],ix+box[4],yres,xres)], border_comp[IdiToIdx(iz+box[0]-1,iy+box[2],ix+box[4],yres,xres)]))
                elif iz == box[1]-box[0]-1 and IdiToIdx(iz+box[0]+1,iy+box[2],ix+box[4],yres,xres) in border_comp_exist:
                    neighbor_label_set_border.add((border_comp[IdiToIdx(iz+box[0],iy+box[2],ix+box[4],yres,xres)], border_comp[IdiToIdx(iz+box[0]+1,iy+box[2],ix+box[4],yres,xres)]))
                else: #TODO get rif of 100000000 and replace by highest number
                    neighbor_label_set_border.add((border_comp[IdiToIdx(iz+box[0],iy+box[2],ix+box[4],yres,xres)], 100000000))

    for iz in range(0, box[1]-box[0]):
        for iy in [0, box[3]-box[2]-1]:
            for ix in range(0, box[5]-box[4]):

                if iy == 0 and IdiToIdx(iz+box[0],iy+box[2]-1,ix+box[4],yres,xres) in border_comp_exist:
                    neighbor_label_set_border.add((border_comp[IdiToIdx(iz+box[0],iy+box[2],ix+box[4],yres,xres)], border_comp[IdiToIdx(iz+box[0],iy+box[2]-1,ix+box[4],yres,xres)]))
                elif iy == box[3]-box[2]-1 and IdiToIdx(iz+box[0],iy+box[2]+1,ix+box[4],yres,xres) in border_comp_exist:
                    neighbor_label_set_border.add((border_comp[IdiToIdx(iz+box[0],iy+box[2],ix+box[4],yres,xres)], border_comp[IdiToIdx(iz+box[0],iy+box[2]+1,ix+box[4],yres,xres)]))
                else:
                    neighbor_label_set_border.add((border_comp[IdiToIdx(iz+box[0],iy+box[2],ix+box[4],yres,xres)], 100000000))
    for iz in range(0, box[1]-box[0]):
        for iy in range(0, box[3]-box[2]):
            for ix in [0, box[5]-box[4]-1]:

                if ix == 0 and IdiToIdx(iz+box[0],iy+box[2],ix+box[4]-1,yres,xres) in border_comp_exist:
                    neighbor_label_set_border.add((border_comp[IdiToIdx(iz+box[0],iy+box[2],ix+box[4],yres,xres)], border_comp[IdiToIdx(iz+box[0],iy+box[2],ix+box[4]-1,yres,xres)]))
                elif ix == box[5]-box[4]-1 and IdiToIdx(iz+box[0],iy+box[2],ix+box[4]+1,yres,xres) in border_comp_exist:
                    neighbor_label_set_border.add((border_comp[IdiToIdx(iz+box[0],iy+box[2],ix+box[4],yres,xres)], border_comp[IdiToIdx(iz+box[0],iy+box[2],ix+box[4]+1,yres,xres)]))
                else:
                    neighbor_label_set_border.add((border_comp[IdiToIdx(iz+box[0],iy+box[2],ix+box[4],yres,xres)], 100000000))


    return neighbor_label_set_border

# find sets of adjacent components TODO do not pass border_comp and border_comp_exist (why?)
@njit
def findAdjLabelSetLocal(box, cc_labels, border_comp, border_comp_exist, yres, xres):

    neighbor_label_set_inside = set()
    neighbor_label_set_border = set()

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
                border_comp[IdiToIdx(iz+box[0],iy+box[2],ix+box[4],yres,xres)] = cc_labels[iz,iy,ix]
                border_comp_exist.add(IdiToIdx(iz+box[0],iy+box[2],ix+box[4],yres,xres))

                neighbor_label_set_border.add((cc_labels[iz,iy,ix], 100000000))

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

                border_comp[IdiToIdx(iz+box[0],iy+box[2],ix+box[4],yres,xres)] = cc_labels[iz,iy,ix]
                border_comp_exist.add(IdiToIdx(iz+box[0],iy+box[2],ix+box[4],yres,xres))

                neighbor_label_set_border.add((cc_labels[iz,iy,ix], 100000000))

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

                border_comp[IdiToIdx(iz+box[0],iy+box[2],ix+box[4],yres,xres)] = cc_labels[iz,iy,ix]
                border_comp_exist.add(IdiToIdx(iz+box[0],iy+box[2],ix+box[4],yres,xres))

                neighbor_label_set_border.add((cc_labels[iz,iy,ix], 100000000))

    return neighbor_label_set_inside, neighbor_label_set_border, border_comp, border_comp_exist

# set dict value undetermined set to 0
def setUndeterminedtoNonHole(undetermined, associated_label):

    for _ in range(len(undetermined)):

        elem = undetermined.pop()
        associated_label[elem] = 0

    if len(undetermined)>0:
        raise ValueError("Uknown Error")

    return associated_label

# get neighbor label dict from neighbor label set
def writeNeighborLabelDict(neighbor_label_set):

    neighbor_label_dict = dict()
    # time_start = time.time()

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

    # print("time to get neighbor_label_dict dict: " + str(time.time()-time_start))

    return neighbor_label_dict

# create string of connected components that are a whole
def findAssociatedLabels(neighbor_label_dict, undetermined, associated_label):

    # time_start = time.time()
    border_contact = set()

    while len(undetermined)>0:

        query_comp = undetermined.pop()

        #check if it has only one neighbor and this neighbor is a neuron
        if len(neighbor_label_dict[query_comp])==1 and neighbor_label_dict[query_comp][0]!=100000000 and neighbor_label_dict[query_comp][0]>0:
            associated_label[query_comp] = neighbor_label_dict[query_comp][0]

        # otherwise unroll neighbors to identify
        else:

            # list of nodes to expand
            open = []
            # iterate over all neighbors and add them to the open set, if they are a background componente (i.e. are negative)
            for elem in neighbor_label_dict[query_comp]:
                if elem == 100000000:
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
                if elem == 100000000:
                    if 100000000 not in neighbor_label_dict[query_comp]:
                        neighbor_label_dict[query_comp].append(100000000)
                else:
                    for son in neighbor_label_dict[elem]:
                        if son not in neighbor_label_dict[query_comp]:
                            neighbor_label_dict[query_comp].append(son)
                            if son<0:
                                open.insert(0,son)

            # check if there is a bordercontact, then add to bordercontact but remove all elemnts from undetermined (will be added again later)
            if 100000000 in neighbor_label_dict[query_comp]:
                border_contact.add(query_comp)
                for elem in neighbor_label_dict[query_comp]:
                    if elem < 0:
                        border_contact.add(elem)
                        undetermined.discard(elem)

            # if component does not have border contact, it can now be definitley determined if it is a hole or not
            else:
            # check again if there is only one positive neighbor and that it is not boundary and it is a neuron, if so, it is a hole #TODO: second check could be removed
                if len(list(filter(lambda a: a>0, neighbor_label_dict[query_comp])))==1 and np.max(neighbor_label_dict[query_comp])>0:
                    associated_label[query_comp] = np.max(neighbor_label_dict[query_comp])
                    for elem in neighbor_label_dict[query_comp]:
                        if elem < 0:
                            associated_label[elem]=np.max(neighbor_label_dict[query_comp])
                            undetermined.discard(elem)
                else:
                    associated_label[query_comp] = 0
                    for elem in neighbor_label_dict[query_comp]:
                        if elem < 0:
                            associated_label[elem]=0
                            undetermined.discard(elem)
            # delte open set
            del open

    # print("time to get associated label dict: " + str(time.time()-time_start))

    if len(undetermined)>0:
        raise ValueError("Unknown Error")

    undetermined = undetermined.union(border_contact)

    return associated_label, undetermined

# fill detedted wholes and give non_wholes their ID (for visualization)
def fillWholes(output_path,bz,by,bx,associated_label):

    # create filename
    input_name = "cc_labels"+"_z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)
    box = getBoxAll(output_path+input_name+".h5")

    # read in data
    cc_labels = readData(box, output_path+input_name+".h5")

    # use nopython to do actual computation
    cc_labels = fillwholesNoPython(box,cc_labels,associated_label)

    output_name = "block_filled"+"_z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)
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
        print(str("Processing file " + sample_name).format(sample_name), end='\r')
        if i is slices_s:
            labels_concat = readData(box, data_path+sample_name+".h5")
        else:
            labels_temp = readData(box, data_path+sample_name+".h5")
            labels_old = labels_concat.copy()
            del labels_concat
            labels_concat = np.concatenate((labels_old,labels_temp),axis=0)
            del labels_temp

    print("Concat size/ shape: " + str(labels_concat.nbytes) + '/ ' + str(labels_concat.shape))
    writeData(output_path, labels_concat)

    del labels_concat

def concatBlocks(z_start, n_blocks_z, n_blocks_y, n_blocks_x, output_path):

    for bz in range(z_start, z_start+n_blocks_z):
        print("processing z block " + str(bz))
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):

                input_name = "block_filled"+"_z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)

                if bz==z_start and by==bx==0:
                    box = getBoxAll(filename=output_path+input_name+".h5")
                    bs_z = box[1]
                    bs_y = box[3]
                    bs_x = box[5]
                    labels_concat =  np.zeros((bs_z*n_blocks_z,bs_y*n_blocks_y,bs_x*n_blocks_x),dtype=np.uint16)
                    print(labels_concat.shape)

                if box != getBoxAll(filename=output_path+input_name+".h5"):
                    raise ValueError("Cannot Concat Blocks of different size!")
                else:
                    labels_concat[(bz-z_start)*bs_z:((bz-z_start)+1)*bs_z,by*bs_y:(by+1)*bs_y,bx*bs_x:(bx+1)*bs_x] = readData(box=[1], filename=output_path+input_name+".h5")

    print("Concat size/ shape: " + str(labels_concat.nbytes) + '/ ' + str(labels_concat.shape))
    output_name = "filled"
    writeData(output_path+output_name, labels_concat)

    return labels_concat

def evaluateWholes(folder_path, ID_A, ID_B, sample_name):
    print("Evaluating wholes...")
    # load gt wholes
    gt_wholes_filepath = folder_path+"/"+ID_A+"/"+"wholes"+".h5"
    box = getBoxAll(gt_wholes_filepath)
    wholes_gt = readData(box, gt_wholes_filepath)

    # load block wholes
    inBlocks_wholes_filepath = folder_path+"/"+ID_B+"/"+"wholes"+".h5"
    box = getBoxAll(inBlocks_wholes_filepath)
    wholes_inBlocks = readData(box, inBlocks_wholes_filepath)

    try:# check that both can be converted to int16
        if np.max(wholes_gt)>32767 or np.max(wholes_inBlocks)>32767:
            raise ValueError("Cannot convert wholes to int16 (max is >32767)")
    except:
        print("Cannot convert wholes to int16 (max is >32767) -  ignored this Error")

    wholes_gt = wholes_gt.astype(np.int16)
    wholes_inBlocks = wholes_inBlocks.astype(np.int16)
    wholes_gt = np.subtract(wholes_gt, wholes_inBlocks)
    diff = wholes_gt
    # free some RAM
    del wholes_gt, wholes_inBlocks

    print("Freed memory")

    if np.min(diff)<0:
        FP = diff.copy()
        FP[FP>0]=0
        n_points_FP = np.count_nonzero(FP)
        n_comp_FP = computeConnectedComp26(FP)-1
        print("FP classifications (points/components): " + str(n_points_FP) + "/ " +str(n_comp_FP))

        # unique_values = np.unique(FP)
        # for u in unique_values:
        #     if u!=0:
        #         print("Coordinates of component " + str(u))
        #         coods = np.argwhere(FP==u)
        #         for i in range(coods.shape[0]):
        #             print(str(coods[i,0]) + ", " + str(coods[i,1]) + ", " + str(coods[i,2]))

        del FP
    else:
        print("No FP classification")

    if np.max(diff)>0:
        FN = diff.copy()
        FN[FN<0]=0
        n_points_FN = np.count_nonzero(FN)
        n_comp_FN = computeConnectedComp26(FN)-1
        print("FN classifications (points/components): " + str(n_points_FN) + "/ " +str(n_comp_FN))
        del FN

    else:
        print("No FN classification")

    output_name = 'diff_wholes'
    writeData(folder_path+"/"+output_name, diff)

    del diff

@njit
def IdxToIdi(iv, yres, xres):
    iz = iv // (yres * xres)
    iy = (iv - iz * yres * xres) // xres
    ix = iv % xres
    return iz, iy, ix

@njit
def IdiToIdx(ix, iy, iz, yres, xres):
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
            print(str("Processing file " + sample_name).format(sample_name), end='\r')
            if i is self.slices_start:
                labels_concat = readData(self.box_concat, self.data_path+sample_name+".h5")
            else:
                labels_temp = readData(self.box_concat, self.data_path+sample_name+".h5")
                labels_old = labels_concat.copy()
                del labels_concat
                labels_concat = np.concatenate((labels_old,labels_temp),axis=0)
                del labels_temp

        print("Concat size/ shape: " + str(labels_concat.nbytes) + '/ ' + str(labels_concat.shape))
        writeData(self.folder_path+self.sample_name, labels_concat)

        del labels_concat

    def evaluateWholes(self, ID_A, ID_B):
        print("Evaluating wholes...")
        # load gt wholes
        gt_wholes_filepath = self.folder_path+"/"+ID_A+"/"+"wholes"+".h5"
        box = getBoxAll(gt_wholes_filepath)
        wholes_gt = readData(box, gt_wholes_filepath)

        # load block wholes
        inBlocks_wholes_filepath = self.folder_path+"/"+ID_B+"/"+"wholes"+".h5"
        box = getBoxAll(inBlocks_wholes_filepath)
        wholes_inBlocks = readData(box, inBlocks_wholes_filepath)

        try:# check that both can be converted to int16
            if np.max(wholes_gt)>32767 or np.max(wholes_inBlocks)>32767:
                raise ValueError("Cannot convert wholes to int16 (max is >32767)")
        except:
            print("Cannot convert wholes to int16 (max is >32767) -  ignored this Error")

        wholes_gt = wholes_gt.astype(np.int16)
        wholes_inBlocks = wholes_inBlocks.astype(np.int16)
        wholes_gt = np.subtract(wholes_gt, wholes_inBlocks)
        diff = wholes_gt
        # free some RAM
        del wholes_gt, wholes_inBlocks

        print("Freed memory")

        if np.min(diff)<0:
            FP = diff.copy()
            FP[FP>0]=0
            n_points_FP = np.count_nonzero(FP)
            n_comp_FP = computeConnectedComp26(FP)-1
            print("FP classifications (points/components): " + str(n_points_FP) + "/ " +str(n_comp_FP))

            # unique_values = np.unique(FP)
            # for u in unique_values:
            #     if u!=0:
            #         print("Coordinates of component " + str(u))
            #         coods = np.argwhere(FP==u)
            #         for i in range(coods.shape[0]):
            #             print(str(coods[i,0]) + ", " + str(coods[i,1]) + ", " + str(coods[i,2]))

            del FP
        else:
            print("No FP classification")

        if np.max(diff)>0:
            FN = diff.copy()
            FN[FN<0]=0
            n_points_FN = np.count_nonzero(FN)
            n_comp_FN = computeConnectedComp26(FN)-1
            print("FN classifications (points/components): " + str(n_points_FN) + "/ " +str(n_comp_FN))
            del FN

        else:
            print("No FN classification")

        output_name = 'diff_wholes_'+ID_A+"_"+ID_B
        writeData(self.folder_path+"/"+ID_B+"/"+output_name, diff)

        del diff

    def readLabelsZebrafinch(self, data_path, sample_name, slice, bz, by, bx, bs_z, bs_y, bs_x):
        filename = data_path+"/"+sample_name+"/"+str(bz*128).zfill(4)
        box = [0, bs_z, 0, bs_y, 0, bs_x]
        #TODO: change this to always read entire chunk and then check size, also change the damn ".h5" supplement studpid
        self.labels_in = readData(box, filename+".h5")
        self.bs_z = bs_z
        self.bs_y = bs_y
        self.bs_x = bs_x
        self.bz=bz
        self.by=by
        self.bx=bx

    def computeStepOne(self, label_start, max_labels_block, output_path):

        # TODO get rid of this
        box = [self.bz*self.bs_z,(self.bz+1)*self.bs_z,self.by*self.bs_y,(self.by+1)*self.bs_y,self.bx*self.bs_x,(self.bx+1)*self.bs_x]

        border_comp_local = Dict.empty(key_type=types.int64,value_type=types.int64)
        border_comp_exist_local = {(2**30)}
        associated_label_local = Dict.empty(key_type=types.int64,value_type=types.int64)

        cc_labels, n_comp = computeConnectedComp6(self.labels_in,label_start,max_labels_block)

        del self.labels_in

        output_name = "cc_labels"+"_z"+str(self.bz).zfill(4)+"y"+str(self.by).zfill(4)+"x"+str(self.bx).zfill(4)
        writeData(output_path+output_name, cc_labels)

        neighbor_label_set_inside_local, neighbor_label_set_border_local, border_comp_local, border_comp_exist_local = findAdjLabelSetLocal(
                                                                            box, cc_labels, border_comp_local, border_comp_exist_local, self.yres, self.xres)

        del cc_labels

        neighbor_label_set = neighbor_label_set_inside_local.union(neighbor_label_set_border_local)
        neighbor_label_dict = writeNeighborLabelDict(neighbor_label_set)

        undetermined_local = set(neighbor_label_dict.keys())
        associated_label_local, undetermined_local = findAssociatedLabels(neighbor_label_dict=neighbor_label_dict, undetermined=undetermined_local, associated_label=associated_label_local)

        del neighbor_label_set, neighbor_label_set_border_local, neighbor_label_dict

        self.n_comp=n_comp

        output_name = "_z"+str(self.bz).zfill(4)+"y"+str(self.by).zfill(4)+"x"+str(self.bx).zfill(4)

        dumpNumbaDictToFile(border_comp_local, "border_comp_local", output_path, output_name)
        dumpToFile(border_comp_exist_local, "border_comp_exist_local", output_path, output_name)
        dumpToFile(neighbor_label_set_inside_local, "neighbor_label_set_inside_local", output_path, output_name)
        dumpNumbaDictToFile(associated_label_local, "associated_label_local", output_path, output_name)
        dumpToFile(undetermined_local, "undetermined_local", output_path, output_name)

    def setRes(self, zres,yres,xres):
        self.zres = zres
        self.yres = yres
        self.xres = xres

def compareOutp(output_path, sample_name, ID_B):
    vizWholes = True

    blockA = dataBlock(viz_wholes=vizWholes)

    # blockA.createNewBlock(  data_path="/home/frtim/wiring/raw_data/segmentations/Zebrafinch/",
    #                         output_path=output_path,
    #                         slices_start=2,
    #                         slices_end=3,
    #                         box_concat=[0,128,0,256,0,256])
    #
    # blockA.concatFiles()
    #
    # blockA.processFile(ID="gt",rel_block_size=1)

    blockA.useExistingFolder(output_path=output_path, sample_name=sample_name)

    # blockA.processFile(ID=ID_B,rel_block_size=0.25)

    blockA.evaluateWholes(ID_A="gt", ID_B=ID_B)


def main():

    output_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/"
    data_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/"
    sample_name = "ZF_concat_2to4_0512_0512"
    outp_ID = "update2"

    output_path = data_path + "/" + sample_name + "/" + outp_ID + "/"
    if os.path.exists(output_path):
        raise ValueError("Folderpath " + data_path + " already exists!")
    else:
        os.mkdir(output_path)

    # start slice of zebrafinch block
    slice_start = 2

    # compute number of blocks and block size
    bs_z = 128
    n_blocks_z = 3
    bs_y = 512
    n_blocks_y = 1
    bs_x = 512
    n_blocks_x = 1

    zres=bs_z*n_blocks_z
    yres=bs_y*n_blocks_y
    xres=bs_x*n_blocks_x

    #counters
    cell_counter = 0
    n_comp_total = 0
    label_start = -1

    max_labels_block = bs_z*bs_y*bs_x

    # STEP 1
    for bz in range(slice_start, slice_start+n_blocks_z):
        print("processing z block " + str(bz))
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):

                currBlock = dataBlock(viz_wholes=True)
                currBlock.readLabelsZebrafinch(data_path=data_path, sample_name=sample_name, slice=bz, bz=bz, by=by, bx=bx, bs_z=bs_z, bs_y=bs_y, bs_x=bs_x)
                currBlock.setRes(zres=zres,yres=yres,xres=xres)

                #TODO get rid off max labels block, just compute it in the connectedcomp function as the size of passed block
                currBlock.computeStepOne(label_start=label_start, max_labels_block=max_labels_block, output_path=output_path)

                cell_counter += 1
                n_comp_total += currBlock.n_comp
                label_start = label_start-max_labels_block

                del currBlock

    print("n_comp_total: " + str(n_comp_total))

    # STEP 2
    border_comp_global = Dict.empty(key_type=types.int64,value_type=types.int64)
    border_comp_exist_global = {(2**30)}
    neighbor_label_set_inside_global = set()
    associated_label_global = Dict.empty(key_type=types.int64,value_type=types.int64)
    undetermined_global = set()

    for bz in range(slice_start, slice_start+n_blocks_z):
        print("processing z block " + str(bz))
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):

                output_name = "_z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)

                border_comp_local = readFromFile("border_comp_local", output_path, output_name)
                border_comp_exist_local = readFromFile("border_comp_exist_local", output_path, output_name)
                neighbor_label_set_inside_local = readFromFile("neighbor_label_set_inside_local", output_path, output_name)
                associated_label_local = readFromFile("associated_label_local", output_path, output_name)
                undetermined_local = readFromFile("undetermined_local", output_path, output_name)

                border_comp_global.update(border_comp_local)
                border_comp_exist_global = border_comp_exist_global.union(border_comp_exist_local)
                neighbor_label_set_inside_global = neighbor_label_set_inside_global.union(neighbor_label_set_inside_local)
                associated_label_global.update(associated_label_local)
                undetermined_global = undetermined_global.union(undetermined_local)

                del border_comp_local, border_comp_exist_local, neighbor_label_set_inside_local, associated_label_local, undetermined_local

    border_comp_exist_global.remove((2**30))
    neighbor_label_set_border_global = {(1,1)}

    counter_total = 0

    for bz in range(slice_start, slice_start+n_blocks_z):
        print("processing z block " + str(bz))
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):

                box = [bz*bs_z,(bz+1)*bs_z,by*bs_y,(by+1)*bs_y,bx*bs_x,(bx+1)*bs_x]
                # print(box)

                neighbor_label_set_border_global = findAdjLabelSetGlobal(box, neighbor_label_set_border_global,
                                                        border_comp_global, border_comp_exist_global, yres, xres)

    neighbor_label_set_border_global.remove((1,1))
    neighbor_label_set = neighbor_label_set_inside_global.union(neighbor_label_set_border_global)

    print("Find associated labels...")
    neighbor_label_dict = writeNeighborLabelDict(neighbor_label_set)
    associated_label_global, undetermined_global = findAssociatedLabels(neighbor_label_dict, undetermined_global, associated_label_global)
    associated_label_global = setUndeterminedtoNonHole(undetermined_global, associated_label_global)

    output_name = ""
    dumpNumbaDictToFile(associated_label_global, "associated_label_global", output_path, output_name)

    del associated_label_global

    # STEP 3
    oytput_name = ""
    associated_label_global = Dict.empty(key_type=types.int64,value_type=types.int64)
    associated_label_global.update(readFromFile("associated_label_global", output_path, output_name))

    print("Fill wholes...")
    # process blocks by iterating over all bloks
    for bz in range(slice_start, slice_start+n_blocks_z):
        print("processing z block " + str(bz))
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):

                fillWholes(output_path=output_path,bz=bz,by=by,bx=bx,associated_label=associated_label_global)

    # (STEP 4 visualize wholes
    # print out total of found wholes
    blocks_concat = concatBlocks(z_start=slice_start, n_blocks_z=n_blocks_z, n_blocks_y=n_blocks_y, n_blocks_x=n_blocks_x, output_path=output_path)

    filename = data_path+"/"+sample_name+"/"+sample_name+".h5"
    box = getBoxAll(filename)
    labels_inp = readData(box, filename)
    neg = np.subtract(blocks_concat, labels_inp)
    output_name = "wholes"
    writeData(output_path+output_name, neg)

    compareOutp(output_path=data_path,sample_name=sample_name,ID_B=outp_ID )

if __name__== "__main__":
  main()
