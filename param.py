import math

# Header file for block processing, includes parameters and paths
isCluster = False

if isCluster:
    output_path = "/n/home12/tfranzmeyer/wiring/raw_data/segmentations/Zebrafinch/"
    data_path = "/n/home12/tfranzmeyer/wiring/raw_data/segmentations/Zebrafinch/"
    sample_name = ""

    # compute number of blocks and block size
    bs_z = 128
    n_blocks_z = 20
    bs_y = 5456
    n_blocks_y = 1
    bs_x = 5332
    n_blocks_x = 1

    # start slice of zebrafinch block
    z_start = 0
    y_start = 0
    x_start = 0


if not isCluster:
    output_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/"
    data_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/"
    # sample_name = "ZF_concat_6to7_0512_0512"
    sample_name = "ZF_concat_2to15_0256_0256"

    # compute number of blocks and block size
    bs_z = 128
    n_blocks_z = 14
    bs_y = 128
    n_blocks_y = 2
    bs_x = 128
    n_blocks_x = 2

    # this has to set here and in the bash script
    z_start = 2
    y_start = 0
    x_start = 0

outp_ID = "fresh2"

folder_path = data_path + sample_name + "/" + outp_ID + "/"

#have to creat these in advance
error_path_preparation = data_path + sample_name + "/" + "error_files_preparation/"
output_path_preparation = data_path + sample_name + "/" + "output_files_preparation/"

#these are created by the preparations folder
error_path = folder_path+"error_files/"
output_path = folder_path+"output_files/"

#create file no save n_comp (number of cc3d components) for each procesed block
n_comp_filepath = folder_path+"n_comp.txt"

#memory need per block (in MB)
memory_needed = int(1.1*bs_z*bs_y*bs_x*(8+8+2)/1000/1000)

zres=bs_z*n_blocks_z
yres=bs_y*n_blocks_y
xres=bs_x*n_blocks_x

max_labels_block = bs_z*bs_y*bs_x

#step Two iterations needed:
iterations_needed  = math.ceil(math.log(n_blocks_z)/math.log(2))
