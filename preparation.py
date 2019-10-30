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
makeFolder(param.error_path)
makeFolder(param.output_path)
makeFolder(param.code_run_path)

shutil.copyfile(param.code_path+"param.py", param.code_run_path+"param.py")
shutil.copyfile(param.code_path+"StepOne.py", param.code_run_path+"StepOne.py")
shutil.copyfile(param.code_path+"StepTwoA.py", param.code_run_path+"StepTwoA.py")
shutil.copyfile(param.code_path+"StepTwoB.py", param.code_run_path+"StepTwoB.py")
shutil.copyfile(param.code_path+"StepThree.py", param.code_run_path+"StepThree.py")

print("Created output folders and code folder at " + param.code_run_path)
