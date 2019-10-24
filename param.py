import math

# this exact code was used to computye Zebrafinch ground turth, which is saved (read-only)
# in the folder /n/pfister_lab2/Lab/tfranzmeyer/Zebrafinch/allFilesFullSize/seg_filled


# Header file for block processing, includes parameters and paths
isCluster = False

if isCluster:
    output_path = "/n/pfister_lab2/Lab/tfranzmeyer/Zebrafinch/"
    data_path = "/n/pfister_lab2/Lab/tfranzmeyer/Zebrafinch/"
    sample_name = ""

    # compute number of blocks and block size
    bs_z = 128
    bs_z_last = 68
    n_blocks_z = 45

    bs_y = 5456
    n_blocks_y = 1

    bs_x = 5332
    n_blocks_x = 1

    # start slice of zebrafinch block
    z_start = 0
    y_start = 0
    x_start = 0

    # compute number of blocks and block size
    #bs_z = 128
    #n_blocks_z = 14
    #bs_y = 128
    #n_blocks_y = 2
    #bs_x = 128
    #n_blocks_x = 2

    # this has to set here and in the bash script
    #z_start = 2
    #y_start = 0
    #x_start = 0

if not isCluster:
    output_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/"
    data_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/"
    # sample_name = "ZF_concat_6to7_0512_0512"
    sample_name = "ZF_concat_2to15_0256_0256"

    # compute number of blocks and block size
    bs_z = 128
    bs_z_last = 128
    n_blocks_z = 14

    bs_y = 128
    n_blocks_y = 2

    bs_x = 128
    n_blocks_x = 2

    # this has to set here and in the bash script
    z_start = 2
    y_start = 0
    x_start = 0

outp_ID = "PrintTest"

folder_path = data_path + sample_name + "/" + outp_ID + "/"

# have to creat these in advance
error_path_preparation = data_path + sample_name + "/" + "error_files_preparation/"
output_path_preparation = data_path + sample_name + "/" + "output_files_preparation/"

# these are created by the preparations folder
error_path = folder_path+"error_files/"
output_path = folder_path+"output_files/"

# create file no save n_comp (number of cc3d components) for each procesed block
n_comp_filepath         = folder_path+"n_comp.txt"
step01_timing_filepath  = folder_path+"step01_timing.txt"
step02A_timing_filepath = folder_path+"step02A_timing.txt"
step02B_timing_filepath = folder_path+"step02B_timing.txt"
step03_timing_filepath  = folder_path+"step03_timing.txt"

# memory need per block (in MB)
memory_needed = 80000 #int(1.1*bs_z*bs_y*bs_x*(8+8+2)/1000/1000)

zres=bs_z*n_blocks_z
yres=bs_y*n_blocks_y
xres=bs_x*n_blocks_x

max_labels_block = bs_z*bs_y*bs_x

#step Two iterations needed:
iterations_needed  = math.ceil(math.log(n_blocks_z)/math.log(2))
