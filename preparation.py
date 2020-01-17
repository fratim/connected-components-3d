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
makeFolder(param.folder_path)
makeFolder(param.output_path_filled_segments)
makeFolder(param.output_path_neuron_surfaces)
makeFolder(param.error_path)
makeFolder(param.output_path)
makeFolder(param.slurm_path)

print("Created output folders")
