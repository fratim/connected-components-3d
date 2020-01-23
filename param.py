import math

# Header file for block processing, includes parameters and paths
compute_statistics = True

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

outp_ID = "pad_dis_stat"

folder_path = data_path + sample_name + "/" + outp_ID + "/"
output_path_filled_segments = folder_path+"output_segments/"
output_path_neuron_surfaces = folder_path+"neuron_surfaces/"

# folder to which the current code is copied and where it is run from
code_run_path = "/n/home12/tfranzmeyer/Code/connected-components-3d/"
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

points_per_component_filepath  = folder_path+"points_per_component"
hole_components_filepath  = folder_path+"hole_components"
component_equivalences_filepath  = folder_path+"component_equivalences"

# memory need per block (in MB)
memory_step01_number = int(1.1*max_bs_z*max_bs_y*max_bs_x*(8+8+8)/1000/1000)
memory_step00 = str(memory_step01_number)
memory_step01 = str(memory_step01_number)
memory_step02A = str(int(memory_step01_number*0.05))
memory_step02B = str(int(memory_step01_number*0.5))
memory_step03 = str(int(memory_step01_number*1))

run_hours = str(2)

max_labels_block = max_bs_z*max_bs_y*max_bs_x
