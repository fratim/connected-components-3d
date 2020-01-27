import cc3d
import numpy as np
import time
import h5py
from numba import njit, types
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from numba.typed import Dict
import os
import pickle
import param
import sys
import shutil

from functions import makeFolder


# make these folders and give error if they exist
makeFolder(param.output_path_filled_segments)
makeFolder(param.output_path_filled_segments+param.prefix+"/")

makeFolder(param.output_path_neuron_surfaces)
makeFolder(param.output_path_neuron_surfaces+param.prefix+"/")

makeFolder(param.folder_path)
makeFolder(param.output_path_neuron_surfaces)
makeFolder(param.error_path)
makeFolder(param.output_path)
makeFolder(param.slurm_path)
makeFolder(param.points_per_component_folder)
makeFolder(param.hole_components_folder)
makeFolder(param.component_equivalences_folder)
makeFolder(param.total_times_folder)
makeFolder(param.info_folder)

print("Created output folders")
