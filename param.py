import math

# Header file for block processing, includes parameters and paths
compute_statistics = False
prefix = "Zebrafinch"

# output_path = "/n/pfister_lab2/Lab/tfranzmeyer/Zebrafinch/original_data/"
data_path = "/n/pfister_lab2/Lab/tfranzmeyer/Data/1024x1024x1024/1_discarded_1024x1024x1024/"
folder_path = "/n/pfister_lab2/Lab/tfranzmeyer/Data/1024x1024x1024/holefilling/"
code_path = "/n/home12/tfranzmeyer/Code/connected-components-3d/"
output_path_filled_segments = "/n/pfister_lab2/Lab/tfranzmeyer/Data/1024x1024x1024/2_discarded_filled_padded_1024x1024x1024/"
output_path_neuron_surfaces = "/n/pfister_lab2/Lab/tfranzmeyer/Data/1024x1024x1024/d_seg_discarded_filled_surfaces/"

# compute number of blocks and block size
max_bs_z = 1024
n_blocks_z = 6

max_bs_y = 1024
n_blocks_y = 6

max_bs_x = 1024
n_blocks_x = 6

# start slice of zebrafinch block
z_start = 0
y_start = 0
x_start = 0


# folder to which the current code is copied and where it is run from
slurm_path = folder_path + "slurm_files/"
# these are created by the preparations folder
error_path = folder_path+"error_files/"
output_path = folder_path+"output_files/"

# create file no save n_comp (number of cc3d components) for each procesed block
info_folder                 = folder_path+"info_files/"
total_times_folder          = folder_path+"total_times/"
n_comp_filepath             = info_folder+"n_comp.txt"
step01_info_filepath        = info_folder+"step01_info.txt"
step02A_info_filepath       = info_folder+"step02A_info.txt"
step02B_info_filepath       = info_folder+"step02B_info.txt"
step03_info_filepath        = info_folder+"step03_info.txt"
total_time_filepath         = total_times_folder+"total_times"

points_per_component_folder  = folder_path+"points_per_component/"
hole_components_folder  = folder_path+"hole_components/"
component_equivalences_folder  = folder_path+"component_equivalences/"

points_per_component_filepath  = points_per_component_folder+"points_per_component"
hole_components_filepath  = hole_components_folder+"hole_components"
component_equivalences_filepath  = component_equivalences_folder+"/component_equivalences"

# memory need per block (in MB)
memory_step01_number = int(1.1*max_bs_z*max_bs_y*max_bs_x*(8+8+8)/1000/1000)
memory_step00 = str(memory_step01_number)
memory_step01 = str(memory_step01_number)
memory_step02A = str(int(memory_step01_number*0.05))
memory_step02B = str(int(memory_step01_number*0.5))
memory_step03 = str(int(memory_step01_number*1))

run_hours = str(4)

max_labels_block = max_bs_z*max_bs_y*max_bs_x
