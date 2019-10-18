# Header file for block processing, includes parameters and paths

output_path = "/n/home12/tfranzmeyer/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/"
data_path = "/n/home12/tfranzmeyer/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/"
sample_name = "ZF_concat_6to7_0512_0512"

outp_ID = "clusterOP_3"

folder_path = data_path + sample_name + "/" + outp_ID + "/"

#have to creat these in advance
error_path_preparation = data_path + sample_name + "/" + "error_files_preparation/"
output_path_preparation = data_path + sample_name + "/" + "output_files_preparation/"

#these are created by the preparations folder
error_path = folder_path+"error_files/"
output_path = folder_path+"output_files/"


# compute number of blocks and block size
bs_z = 64
n_blocks_z = 4
bs_y = 128
n_blocks_y = 4
bs_x = 128
n_blocks_x = 4

# start slice of zebrafinch block
z_start = 6
y_start = 0
x_start = 0

zres=bs_z*n_blocks_z
yres=bs_y*n_blocks_y
xres=bs_x*n_blocks_x

max_labels_block = bs_z*bs_y*bs_x
