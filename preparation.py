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

from functions import makeFolder


# make these folders and give error if they exist
makeFolder(param.folder_path)
makeFolder(param.error_path)
makeFolder(param.output_path)

#make these folders if they do not exist yet
os.mkdir(param.output_path_preparation)
os.mkdir(param.error_path_preparation)

print("Created output folders")
