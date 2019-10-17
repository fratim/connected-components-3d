# Header file for block processing, includes parameters and paths

output_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/"
data_path = "/home/frtim/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/"
sample_name = "ZF_concat_6to7_0512_0512"
outp_ID = "hallo6"

folder_path = data_path + sample_name + "/" + outp_ID + "/"

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
