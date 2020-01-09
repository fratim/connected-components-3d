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
makeFolder(param.code_run_path)
makeFolder(param.slurm_path)

shutil.copyfile(param.code_path+"param.py", param.code_run_path+"param.py")
shutil.copyfile(param.code_path+"functions.py", param.code_run_path+"functions.py")
shutil.copyfile(param.code_path+"preparation.py", param.code_run_path+"preparation.py")
shutil.copyfile(param.code_path+"createSlurms.py", param.code_run_path+"createSlurms.py")
shutil.copyfile(param.code_path+"stepZero.py", param.code_run_path+"stepZero.py")
shutil.copyfile(param.code_path+"stepOne.py", param.code_run_path+"stepOne.py")
shutil.copyfile(param.code_path+"stepTwoA.py", param.code_run_path+"stepTwoA.py")
shutil.copyfile(param.code_path+"stepTwoB.py", param.code_run_path+"stepTwoB.py")
shutil.copyfile(param.code_path+"stepThree.py", param.code_run_path+"stepThree.py")

print("Created output folders and code folder at " + param.code_run_path)
