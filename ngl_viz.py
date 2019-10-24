import neuroglancer
import numpy as np
import sys
import h5py
import sys

ip='localhost' # or public IP of the machine for sharable display
port=98092 # change to an unused port number
neuroglancer.set_server_bind_address(bind_address=ip,bind_port=port)
viewer=neuroglancer.Viewer()

#read data from HD5, given the file path
def ReadH5File(filename,box):
    # return the first h5 dataset from this file
    with h5py.File(filename, 'r') as hf:
        keys = [key for key in hf.keys()]
        # print("Data keys are: ", str(keys))
        d = hf[keys[0]]
        # print("Data shape: ", str(d.shape))
        # load selected part of data
        if box[0]==1:
            data = np.array(d)
        else:
            data = np.array(d[box[0]:box[1],box[2]:box[3],box[4]:box[5]])
# ,dtype=np.dtype(np.bool_)
    return data

#read data from HD5, given the file path
def readData(box, filename):
    # read in data block
    data_in = ReadH5File(filename, box)

    labels = data_in

    # print("data read in; shape: " + str(data_in.shape) + "; DataType: " + str(data_in.dtype))

    return labels

def loadViz(box, path, caption, res, printIDs, idRes, printCoods):

    print("-----------------------------------------------------------------")
    print ('loading ' + caption + "...")
    gt = readData(box, path)
    print("dtype is: " + str(gt.dtype) + ", shape is: " + str(gt.shape))
    gt = gt.astype(np.uint16)

    if printIDs:
        unique_values = np.unique(gt[::idRes,::idRes,::idRes])
        uniq_txt = np.expand_dims(unique_values,axis=1).transpose()
        np.savetxt(sys.stdout.buffer, uniq_txt, delimiter=',', fmt='%d')

    if printCoods:
        for u in unique_values:
            if u!=0:
                print("Coordinates of component " + str(u))
                coods = np.argwhere(gt==u)
                print(coods)
                # for i in int(coods.shape[0]):
                #     print(str(coods[i,0]) + ", " + str(coods[i,1]) + ", " + str(coods[i,2]))

    with viewer.txn() as s:
        s.layers.append(
            name=caption,
            layer=neuroglancer.LocalVolume(
                data=gt,
                voxel_size=res,
                volume_type='segmentation'
            ))

idRes = 1 #which resolution to use to search for IDs
res=[20,18,18]; # resolution of the data
data_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/"
sample_name= "ZF_concat_2to15_0256_0256"
compare= "testwalls"

# box = [0,1152,0,1000,0,1000]
box = [1]

fn_org =                 data_path + sample_name + "/" + sample_name + ".h5"
fn_filled_gt =           data_path + sample_name + "/" + "gt/" + "filled.h5"
fn_wholes_gt =           data_path + sample_name + "/" + "gt/" + "wholes.h5"
fn_filled_compare =      data_path + sample_name + "/" + compare + "/" + "filled.h5"
fn_wholes_compare =      data_path + sample_name + "/" + compare + "/" + "wholes.h5"
fn_diff_wholes_compare = data_path + sample_name + "/" + compare + "/" + "diff_wholes.h5"

print("----------------------------HOST:---------------------------------")
print("-----------------------------------------------------------------")
print(viewer)

loadViz(box=box, path=fn_org,                   caption="original",             res=res, printIDs = True, idRes=4*idRes,    printCoods=False)
loadViz(box=box, path=fn_filled_gt,           caption="filled_gt",            res=res, printIDs = True, idRes=4*idRes,    printCoods=False)
loadViz(box=box, path=fn_wholes_gt,             caption="wholes_gt",            res=res, printIDs = True, idRes=idRes,      printCoods=False)
loadViz(box=box, path=fn_filled_compare,        caption="filled_comp",          res=res, printIDs = True, idRes=idRes,      printCoods=False)
loadViz(box=box, path=fn_wholes_compare,        caption="wholes_comp",          res=res, printIDs = True, idRes=idRes,      printCoods=False)
loadViz(box=box, path=fn_diff_wholes_compare,   caption="diff_wholes_comp",     res=res, printIDs = True, idRes=idRes,      printCoods=True)

print("----------------------------DONE---------------------------------")
print("-----------------------------------------------------------------")
