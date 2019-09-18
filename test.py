import cc3d
import numpy as np
from dataIO import ReadH5File
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import time
from scipy.spatial import distance
import h5py
from numba import njit

global x_start
x_start = 500
global x_end
x_end = 773
global x_size
x_size = x_end-x_start

global y_start
y_start = 1000
global y_end
y_end = 2000
global y_size
y_size = y_end-y_start

global z_start
z_start = 1000
global z_end
z_end = 2000
global z_size
z_size = z_end-z_start

# input lables (0 is background, 1 = neuron1, (2 = neuron2,...))
# defined later:
# global labels

# this function checks if an array of points contains a 2D or a 3D structure
def is2D(points):
    if all_same(points[:,0]) or all_same(points[:,1]) or all_same(points[:,2]):
        return True
    else:
        return False

# function to check if all elements in items have the same value
def all_same(items):
    return all(x == items[0] for x in items)

# fits a convex hull to a 2D point object and returns the coordinates of the points that describe the border
def convHull2D(points):
    if all_same(points[:,0]):
        coods = np.array([points[:,1],points[:,2]]).transpose()
        hull = ConvexHull(coods)
        boundary_idx = np.unique(hull.simplices)
        boundary_pts_coods = points[boundary_idx,:]

    if all_same(points[:,1]):
        coods = np.array([points[:,0],points[:,2]]).transpose()
        hull = ConvexHull(coods)
        boundary_idx = np.unique(hull.simplices)
        boundary_pts_coods = points[boundary_idx,:]

    if all_same(points[:,2]):
        coods = np.array([points[:,0],points[:,1]]).transpose()
        hull = ConvexHull(coods)
        boundary_idx = np.unique(hull.simplices)
        boundary_pts_coods = points[boundary_idx,:]

    return boundary_pts_coods

# fits a convex hull to a 3D point object and returns the coordinates of the points that describe the hull surface
def convHull3D(points):
    hull = ConvexHull(points)
    boundary_idx = np.unique(hull.simplices)
    boundary_pts_coods = points[boundary_idx,:]

    return boundary_pts_coods

# returns the 6 adjacent components for a point, adjacent component is -1 if out of boundary
def getadjcomp(p):

    # set label to -1 if outside of boundary (needed for whole detection)
    comp = np.zeros((1,6))

    # store component number that is adjacend
    if (p[0]+1 < x_size): comp[0,0] = labels[p[0]+1,p[1],p[2]]
    else: comp[0,0] = -1
    if (p[0]-1 > 0): comp[0,1] = labels[p[0]-1,p[1],p[2]]
    else: comp[0,1] = -1
    if (p[1]+1 < y_size): comp[0,2] = labels[p[0],p[1]+1,p[2]]
    else: comp[0,2] = -1
    if (p[1]-1 > 0): comp[0,3] = labels[p[0],p[1]-1,p[2]]
    else: comp[0,3] = -1
    if (p[2]+1 < z_size): comp[0,4] = labels[p[0],p[1],p[2]+1]
    else: comp[0,4] = -1
    if (p[2]-1 > 0): comp[0,5] = labels[p[0],p[1],p[2]-1]
    else: comp[0,5] = -1

    return comp

# checks the adjacent coponents of an array of boundary points and applies rules to check if whole (see code)
def checkifwhole(boundary_pts_coods):

    isWhole = False
    adjComp = np.zeros((6, boundary_pts_coods.shape[0]))
    counter = 0
    connectedNeuron = -1

    for p in boundary_pts_coods:
        adjComp[:,counter] = getadjcomp(p)
        counter = counter + 1

    if -1 in adjComp:
        isWhole = False
        print("Not a Whole, connected to Boundary!")

    elif len(np.unique(adjComp))==2:
        isWhole = True

        #find Neuron that this whole is connected to
        connectedNeuron = np.max(np.absolute(np.unique(adjComp)))
        print("Whole detected! Conntected to Neuron " + str(int(connectedNeuron)))

        #check if whole is composed of Zeros
        if (np.min(np.absolute(np.unique(adjComp)))!= 0):
            isWhole = False
            print("Error! Whole is not composed of 0 and hence is not a valid Whole!")

    elif len(np.unique(adjComp))==1:
        print("Error, this connected component was detected wrong!")

    else:
        print("This connected component is not a whole (>2 neighbors)!")

    return isWhole, connectedNeuron

#read data from HD5, given the file path
def readData(filename):
    # read in data block
    data_in = ReadH5File(filename)

    global labels
    labels = data_in[x_start:x_end,y_start:y_end,z_start:z_end]

    print("data read in; shape: " + str(data_in.shape) + "; DataType: " + str(data_in.dtype) + "; cut to: " + str(labels.shape))

# write data to H5 file
def writeData(filename):
    with h5py.File(filename, 'w') as hf:
        # should cover all cases of affinities/images
        hf.create_dataset("main", data=labels, compression='gzip')

#compute the connected Com ponent labels
def computeConnectedComp():
    lables_inverse = 1 - labels
    connectivity = 6 # only 26, 18, and 6 are allowed
    labels_out = cc3d.connected_components(lables_inverse, connectivity=connectivity)

    # You can extract individual components like so:
    n_comp = np.max(labels_out)
    print("Conntected Regions found: " + str(n_comp))

    # determine indices, numbers and counts for the connected regions
    # unique, counts = np.unique(labels_out, return_counts=True)
    # print("Conntected regions and associated points: ")
    # print(dict(zip(unique, counts)))

    return labels_out, n_comp

# fill a whole by changing the labels to the neuron it belongs to
def fillWhole(coods,connectedNeuron):

    labels[coods[:,0],coods[:,1],coods[:,2]] = np.ones((coods.shape[0],))*(connectedNeuron+1)
    print("Whole has been filled!!")

# find the coordinates of the points that belong to a selected connected component
def findCoodsOfComp(compQuery, compLabels):

    # find coordinates of points that belong to component
    print("executing np.argwhere...")
    idx_comp = np.argwhere(compLabels==compQuery)
    # find coordinates of connected component
    coods = np.array([idx_comp[:,0],idx_comp[:,1],idx_comp[:,2]]).transpose()
    return coods

# find the points that describe the hull space of a given set of points
def findHullPoints(points):
    # check if selected points are in a plane (2D object) and compute points that define hull surface
    if is2D(points):
        HullPts= convHull2D(points)
    else:
        HullPts = convHull3D(points)

    return HullPts

# show points of a cloud in blue and mark the hull points in red
def runViz(coods, hull_coods):
    # debug: plot points as 3D scatter plot, extreme points in red
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coods[:,0],coods[:,1],coods[:,2],c='b')
    ax.scatter(hull_coods[:,0],hull_coods[:,1],hull_coods[:,2],c='r')
    plt.show()

# compute statistics for the connected conmponents that have been found
def doStatistics(isWhole, coods, hull_coods, connectedNeuron, statTable, cnt):

    # account for components starting with 1
    cnt = cnt - 1

    # First column: Check if connected component is a whole
    if isWhole: statTable[cnt,0] = 1
    else: statTable[cnt,1] = 0

    # check if this is a 3D whole (1 if is 3D, otherwise 0)
    if is2D(coods): statTable[cnt,1] = 0
    else: statTable[cnt,2] = 1

    # number of points
    statTable[cnt,3] = int(coods.shape[0])

    # number of hull points
    n_hull_points = int(hull_coods.shape[0])
    statTable[cnt,4] = n_hull_points

    # intermediate step: compute pairwise distances between all hull points
    d_table = distance.cdist(hull_coods, hull_coods, 'euclidean')

    # average distance between hull points
    avg_hull_dist = np.sum(d_table)/(n_hull_points*n_hull_points-n_hull_points)
    statTable[cnt,5] = avg_hull_dist

    # maximum distance between hull points
    max_hull_dist = np.max(d_table)
    statTable[cnt,6] = max_hull_dist

    # intermediate step: find mid point (as the average over all points)
    mid_point = np.mean(coods, axis=0, keepdims=True)

    # intermediate step: find hull mid point (as the average over the hull points)
    hull_mid_point = np.mean(hull_coods, axis=0, keepdims=True)

    # compute mean, median and std for distance for all points from mid point
    d_allPoints_to_mid_table = distance.cdist(mid_point, coods, 'euclidean')
    d_allPoints_to_mid_mean = np.mean(d_allPoints_to_mid_table)
    d_allPoints_to_mid_median = np.median(d_allPoints_to_mid_table)
    d_allPoints_to_mid_std = np.std(d_allPoints_to_mid_table)
    statTable[cnt,7] = d_allPoints_to_mid_mean
    statTable[cnt,8] = d_allPoints_to_mid_median
    statTable[cnt,9] = d_allPoints_to_mid_std

    # compute mean, median and std for distance for all points from hull mid point
    d_allPoints_to_hullmid_table = distance.cdist(hull_mid_point, coods, 'euclidean')
    d_allPoints_to_hullmid_mean = np.mean(d_allPoints_to_hullmid_table)
    d_allPoints_to_hullmid_median = np.median(d_allPoints_to_hullmid_table)
    d_allPoints_to_hullmid_std = np.std(d_allPoints_to_hullmid_table)
    statTable[cnt,10] = d_allPoints_to_hullmid_mean
    statTable[cnt,11] = d_allPoints_to_hullmid_median
    statTable[cnt,12] = d_allPoints_to_hullmid_std

    # compute mean, median and std for distance for hull points from mid point
    d_hullPoints_to_mid_table = distance.cdist(mid_point, hull_coods, 'euclidean')
    d_hullPoints_to_mid_mean = np.mean(d_hullPoints_to_mid_table)
    d_hullPoints_to_mid_median = np.median(d_hullPoints_to_mid_table)
    d_hullPoints_to_mid_std = np.std(d_hullPoints_to_mid_table)
    statTable[cnt,13] = d_hullPoints_to_mid_mean
    statTable[cnt,14] = d_hullPoints_to_mid_median
    statTable[cnt,15] = d_hullPoints_to_mid_std

    # compute mean, median and std for distance for hull points from hull mid point
    d_hullPoints_to_hullmid_table = distance.cdist(hull_mid_point, hull_coods, 'euclidean')
    d_hullPoints_to_hullmid_mean = np.mean(d_hullPoints_to_hullmid_table)
    d_hullPoints_to_hullmid_median = np.median(d_hullPoints_to_hullmid_table)
    d_hullPoints_to_hullmid_std = np.std(d_hullPoints_to_hullmid_table)
    statTable[cnt,16] = d_hullPoints_to_mid_mean
    statTable[cnt,17] = d_hullPoints_to_mid_median
    statTable[cnt,18] = d_hullPoints_to_mid_std

    #distance between hull mid point and all points mid point
    d_mid_to_hullmid = np.linalg.norm(hull_mid_point-mid_point)
    statTable[cnt,19] = d_mid_to_hullmid

    return statTable

# write statistics to a .txt filename
def writeStatistics(statTable, statistics_path, sample_name):
    filename = statistics_path + sample_name.replace("/","_").replace(".","_") + "_statistics_" + str(time.time())[:10] + ".txt"

    header_a = "number,isWhole,is3D,nPoints,nHullPoints,avgHullDist,maxHullDist,"
    header_b = "d_allPoints_to_mid_mean,d_allPoints_to_mid_median,d_allPoints_to_mid_std,"
    header_c = "d_allPoints_to_hullmid_mean,d_allPoints_to_hullmid_median,d_allPoints_to_hullmid_std,"
    header_d = "d_hullPoints_to_mid_mean,d_hullPoints_to_mid_median,d_hullPoints_to_mid_std,"
    header_e = "d_hullPoints_to_hullmid_mean,d_hullPoints_to_hullmid_median,d_hullPoints_to_hullmid_std,"
    header_f = "d_mid_to_hullmid"

    header =  header_a + header_b + header_c + header_d + header_e + header_f

    if(header.count(',')!=(statTable.shape[1]-1)):
        print("Error! Header variables are not equal to number of columns in the statistics!")
    np.savetxt(filename, statTable, delimiter=',', header=header)

@njit(parallel=True)
def findAdjComp(labels_out, n_comp):

    #adj_comp = [[] for _ in range(n_comp)]
    neighbor_sets = set()
    print (x_size)
    print (y_size)
    print (z_size)
    for ix in prange(0, x_size-1):
        for iy in range(0, y_size-1):
            for iz in range(0, z_size-1):
                curr_comp = labels_out[ix,iy,iz]

                if curr_comp != labels_out[ix+1,iy,iz]:
                    neighbor_sets.add((curr_comp, labels_out[ix+1,iy,iz]))
                    neighbor_sets.add((labels_out[ix+1,iy,iz], curr_comp))
                if curr_comp != labels[ix,iy+1,iz]:
                    neighbor_sets.add((curr_comp, labels_out[ix,iy+1,iz]))
                    neighbor_sets.add((labels_out[ix,iy+1,iz], curr_comp))
                if curr_comp != labels[ix,iy,iz+1]:
                    neighbor_sets.add((curr_comp, labels_out[ix,iy,iz+1]))
                    neighbor_sets.add((labels_out[ix,iy,iz+1], curr_comp))

                    #new = np.array(labels[ix+1,iy,iz],dtype=np.uint8)
                    #adj_comp[curr_comp] = np.hstack((adj_comp[curr_comp],new))
                # if labels[ix-1,iy,iz] not in adj_comp[curr_comp]: adj_comp[curr_comp].append(labels[ix-1,iy,iz])
                # if labels[ix,iy+1,iz] not in adj_comp[curr_comp]: adj_comp[curr_comp].append(labels[ix,iy+1,iz])
                # if labels[ix,iy-1,iz] not in adj_comp[curr_comp]: adj_comp[curr_comp].append(labels[ix,iy-1,iz])
                # if labels[ix,iy,iz+1] not in adj_comp[curr_comp]: adj_comp[curr_comp].append(labels[ix,iy,iz+1])
                # if labels[ix,iy,iz-1] not in adj_comp[curr_comp]: adj_comp[curr_comp].append(labels[ix,iy,iz-1])

        # print("ix is " + str(ix) + " of " + str(x_size))
    # print(adj_comp)

    return neighbor_sets

def main():

    # turn Visualization on and off
    Viz = False
    saveStatistics = True
    n_features = 20
    statistics_path = "/home/frtim/wiring/statistics/"
    data_path = "/home/frtim/wiring/raw_data/segmentations/"
    sample_name = "JWR/cell032_downsampled.h5"
    output_name = "JWR/cell032_downsampled_filled_viz.h5"

    # needed to time the code (n_functions as the number of subfunctions considered for timing)

    # read in data (written to global variable labels")
    readData(data_path+sample_name)

    #compute the labels of the conencted connected components
    labels_out, n_comp = computeConnectedComp()

    # check if connected component is a whole)
    # start at 1 as component 0 is always the neuron itself, which has label 1
    # maybe always start at component 2, as this omits the 2 biggest components, whic are normally backgroudn and neuron with labels 0 and 1
    n_start = 2 if Viz else 2

    import time
    start_time = time.time()
    adjComp = findAdjComp(labels_out, n_comp)
    print(adjComp)
    print (time.time() - start_time)

    # if saveStatistics: statTable = np.ones((n_comp-1, n_features))*-1
    # for region in range(n_start,n_comp):
    #
    #     print("Loading component " + str(region) +"...")
    #     # find coordinates of points that belong to the selected component
    #     coods = findCoodsOfComp(region, labels_out)
    #
    #     print("finding points that describe the hull...")
    #     # find coordinates that describe the hull space
    #     hull_coods = findHullPoints(coods)
    #
    #     print("Checking if this is a whole...")
    #     # check if connected component is a whole
    #     isWhole, connectedNeuron = checkifwhole(hull_coods)
    #
    #     print("Felling Whole...")
    #     # fill whole if detected
    #     if isWhole: fillWhole(coods, connectedNeuron)
    #
    #     print("Computing statistics...")
    #     # compute statistics and save to numpy array
    #     if saveStatistics: statTable = doStatistics(isWhole, coods, hull_coods, connectedNeuron, statTable, region)
    #
    #     print("Running Visualization...")
    #     # run visualization
    #     if Viz: runViz(coods,hull_coods)
    #
    # # save the statistics file to a .txt file
    # if saveStatistics: writeStatistics(statTable, statistics_path, sample_name)
    #
    # # write filled data to H5
    # writeData(data_path+output_name)

if __name__== "__main__":
  main()
