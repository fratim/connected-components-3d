import numpy as np
from numba import njit, types
from numba.typed import Dict
import pickle
import time
from functions import computeConnectedComp6, readData, writeData

labels_in = readData([1],"/home/frtim/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/cc_test/ZF_concat_2to5_2048_2048")

max_labels = 128*4*2048*2048
time_start = time.time()
cc_labels, n_comp = computeConnectedComp6(labels_in, -1, max_labels)
print("time needed: " + str(time.time()-time_start))

print("n_comp is: " + str(n_comp))

writeData("/home/frtim/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/cc_test/cc_labels_map", cc_labels)
