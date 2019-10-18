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


makeFolder(param.folder_path)
makeFolder(param.error_path)
makeFolder(param.output_path)

print("Created folder_path, error_path and output_path")
