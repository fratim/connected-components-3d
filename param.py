# Header file for block processing, includes parameters and paths

output_path = "/n/home12/tfranzmeyer/wiring/raw_data/segmentations/Zebrafinch/"
data_path = "/n/home12/tfranzmeyer/wiring/raw_data/segmentations/Zebrafinch/"
sample_name = ""

outp_ID = "cluster_nHoles"

folder_path = data_path + sample_name + "/" + outp_ID + "/"

#have to creat these in advance
error_path_preparation = data_path + sample_name + "/" + "error_files_preparation/"
output_path_preparation = data_path + sample_name + "/" + "output_files_preparation/"

#these are created by the preparations folder
error_path = folder_path+"error_files/"
output_path = folder_path+"output_files/"

#create file no save n_comp (number of cc3d components) for each procesed block
n_comp_filepath = folder_path+"n_comp.txt"

# compute number of blocks and block size
bs_z = 128
n_blocks_z = 4
bs_y = 2048
n_blocks_y = 1
bs_x = 2048
n_blocks_x = 1

# start slice of zebrafinch block
z_start = 2
y_start = 0
x_start = 0

zres=bs_z*n_blocks_z
yres=bs_y*n_blocks_y
xres=bs_x*n_blocks_x

max_labels_block = bs_z*bs_y*bs_x
