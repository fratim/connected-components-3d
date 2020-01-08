import math

# this exact code was used to computye Zebrafinch ground turth, which is saved (read-only)
# in the folder /n/pfister_lab2/Lab/tfranzmeyer/Zebrafinch/allFilesFullSize/seg_filled


# Header file for block processing, includes parameters and paths
isCluster = True
compute_statistics = True

if isCluster:
    output_path = "/n/pfister_lab2/Lab/tfranzmeyer/Zebrafinch/original_data/"
    data_path = "/n/pfister_lab2/Lab/tfranzmeyer/Zebrafinch/original_data/"
    code_path = "/n/home12/tfranzmeyer/Code/connected-components-3d/"
    sample_name = "stacked_256"

    # compute number of blocks and block size
    max_bs_z = 256
    n_blocks_z = 23

    max_bs_y = 2048
    n_blocks_y = 3

    max_bs_x = 2048
    n_blocks_x = 3

    # start slice of zebrafinch block
    z_start = 0
    y_start = 0
    x_start = 0

if not isCluster:
    output_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/"
    data_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/"
    code_path = ""
    sample_name = "ZF_concat_2to14_0256_0256"

    # compute number of blocks and block size
    max_bs_z= 128
    n_blocks_z = 13

    max_bs_y = 128
    n_blocks_y = 2

    max_bs_x = 128
    n_blocks_x = 2

    # this has to set here and in the bash script
    z_start = 2
    y_start = 0
    x_start = 0

outp_ID = "padded_and_statistics"

folder_path = data_path + sample_name + "/" + outp_ID + "/"

# folder to which the current code is copied and where it is run from
code_run_path = folder_path + "code/"
slurm_path = folder_path + "slurm_files/"
# these are created by the preparations folder
error_path = folder_path+"error_files/"
output_path = folder_path+"output_files/"

# create file no save n_comp (number of cc3d components) for each procesed block
n_comp_filepath         = folder_path+"n_comp.txt"
step01_info_filepath  = folder_path+"step01_info.txt"
step02A_info_filepath = folder_path+"step02A_info.txt"
step02B_info_filepath = folder_path+"step02B_info.txt"
step03_info_filepath  = folder_path+"step03_info.txt"

points_per_component_filepath  = folder_path+"points_per_component.txt"
hole_components_filepath  = folder_path+"hole_components.txt"
component_equivalences_filepath  = folder_path+"component_equivalences.txt"

# memory need per block (in MB)
memory_step01_number = int(1.1*max_bs_z*max_bs_y*max_bs_x*(8+8+8)/1000/1000)
memory_step00 = str(memory_step01_number)
memory_step01 = str(memory_step01_number)
memory_step02A = str(int(memory_step01_number*0.05))
memory_step02B = str(int(memory_step01_number*0.5))
memory_step03 = str(int(memory_step01_number*1))

run_hours = str(2)

max_labels_block = max_bs_z*max_bs_y*max_bs_x
