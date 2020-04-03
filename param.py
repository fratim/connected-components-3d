############################################################################################################################################
############################################################################################################################################
##PARAMETERS THAT NEED TO BE SET IN ADVANCE
############################################################################################################################################
############################################################################################################################################
compute_statistics = False
prefix = "Zebrafinch"
cluster_id = "mflow"

# (size_z, size_y, size_x)
blocksize = (128,1024,1024)
volumesize = (6144,6144,6144)

# start slice of zebrafinch block
z_start = 0
y_start = 0
x_start = 0

data_path = "/n/pfister_lab2/Lab/tfranzmeyer/Data/{}x{}x{}_{}/1_discarded_{}x{}x{}/".format(blocksize[2],blocksize[1],blocksize[0],cluster_id,blocksize[2],blocksize[1],blocksize[0])
folder_path = "/n/pfister_lab2/Lab/tfranzmeyer/Data/{}x{}x{}_{}/holefilling/".format(blocksize[2],blocksize[1],blocksize[0],cluster_id)
code_path = "/n/pfister_lab2/Lab/tfranzmeyer/Data/{}x{}x{}_{}/Code/".format(blocksize[2],blocksize[1],blocksize[0],cluster_id)
output_path_filled_segments = "/n/pfister_lab2/Lab/tfranzmeyer/Data/{}x{}x{}_{}/2_discarded_filled_padded_{}x{}x{}/".format(blocksize[2],blocksize[1],blocksize[0],cluster_id,blocksize[2],blocksize[1],blocksize[0])
output_path_neuron_surfaces = "/n/pfister_lab2/Lab/tfranzmeyer/Data/{}x{}x{}_{}/d_seg_discarded_filled_surfaces/".format(blocksize[2],blocksize[1],blocksize[0],cluster_id)

# number of blocks and block size
max_bs_z = blocksize[0]
#n_blocks_z = volumesize[0]//blocksize[0]
n_blocks_z = 8

max_bs_y = blocksize[1]
n_blocks_y = volumesize[1]//blocksize[1]
n_blocks_y = 2

max_bs_x = blocksize[2]
n_blocks_x = volumesize[2]//blocksize[2]
n_blocks_x = 2


############################################################################################################################################
############################################################################################################################################

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
memory_step02B = str(int(memory_step01_number*5))
memory_step02B = str(int(memory_step01_number))
memory_step03 = str(int(memory_step01_number*2))

run_hours = str(2)

max_labels_block = max_bs_z*max_bs_y*max_bs_x
