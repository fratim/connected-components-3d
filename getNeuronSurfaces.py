import glob
import struct
import time
import numpy as np
from numba import njit
import dataIO
import param

OR_Z=0
OR_Y=1
OR_X=2

prefix = "Zebrafinch"

block_size = (param.max_bs_z, param.max_bs_y, param.max_bs_x)
volume_size = (param.max_bs_z*param.n_blocks_z, param.max_bs_y*param.n_blocks_y, param.max_bs_x*param.n_blocks_x)

@njit
def IdentifySurfaces(seg, zblock, yblock, xblock):
    zres, yres, xres = seg.shape
    surfaces = []

    for iz in range(1, zres - 1):
        for iy in range(1, yres - 1):
            for ix in range(1, xres - 1):
                label = seg[iz,iy,ix]
                if not label: continue

                # see if this point belongs to the surface
                surface = False
                if not seg[iz,iy,ix-1]== label: surface = True
                if not seg[iz,iy,ix+1]== label: surface = True
                if not seg[iz,iy-1,ix]== label: surface = True
                if not seg[iz,iy+1,ix]== label: surface = True
                if not seg[iz-1,iy,ix]== label: surface = True
                if not seg[iz+1,iy,ix]== label: surface = True

                if surface:
                	zcoord = zblock * block_size[OR_Z] + iz
                	ycoord = yblock * block_size[OR_Y] + iy
                	xcoord = xblock * block_size[OR_X] + ix
                	index = zcoord * volume_size[OR_Y] * volume_size[OR_X] + ycoord * volume_size[OR_X] + xcoord
                	surfaces.append((label, index))

    return surfaces

surfaces = {}

filenames = sorted(glob.glob("/n/pfister_lab2/Lab/tfranzmeyer/Data/512x512x512/2_discarded_filled_padded_512x512x512/"+prefix+"/*"))

for filename in filenames:
    start_time = time.time()
    seg = dataIO.ReadH5File(filename, [1])
    labels = np.unique(seg)
    print(filename)
    print("shape is: " + str(seg.shape), flush=True)
    for label in labels:
    	if not label: continue

    indices = filename.split('.')[0].split('-')
    zindex = int(indices[2].strip('z'))
    yindex = int(indices[3].strip('y'))
    xindex = int(indices[4].strip('x'))
    surface_points = IdentifySurfaces(seg, zindex, yindex, xindex)

    print("loding file " + filename)
    print("block is " + str(zindex) + ", " + str(yindex) + ", " + str(xindex))

    for (label, index) in surface_points:
    	if not label in surfaces:
    		surfaces[label] = []
    	surfaces[label].append(index)
    print ('Completed {} in {:0.2f} seconds'.format(filename, time.time() - start_time))

for label in sorted(surfaces.keys()):
	output_filename = param.output_path_neuron_surfaces+prefix+"/"+prefix+'-{:06d}.pts'.format(label)
	with open(output_filename, 'wb') as fd:
		fd.write(struct.pack('qqq', volume_size[OR_Z], volume_size[OR_Y], volume_size[OR_X]))
		fd.write(struct.pack('qqq', block_size[OR_Z], block_size[OR_Y], block_size[OR_X]))
		npoints = len(surfaces[label])
		fd.write(struct.pack('qq', label, npoints))
		for surface_point in surfaces[label]:
			fd.write(struct.pack('q', surface_point))
