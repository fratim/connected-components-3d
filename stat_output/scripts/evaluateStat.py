import numpy as np
import math
import glob
import matplotlib.pyplot as plt

holes_of_size = dict()

hole_single = np.genfromtxt("output_files/single_holes.txt", delimiter=',',invalid_raise=True)
hole_multi = np.genfromtxt("output_files/multi_holes.txt", delimiter=',',invalid_raise=True)

hole = np.concatenate((hole_multi,hole_single),axis=0)

total_points = sum(hole)


print("Data read in")
print("Total Holes: " + str(len(hole)))
# print("Holes of size 1: " + str(sum(hole==1)))
# print("Holes of size 2-5: " + str(sum(hole==2)+sum(hole==3)+sum(hole==4)+sum(hole==5)))
# print("Holes of size 6-10: " + str(sum(hole==6)+sum(hole==7)+sum(hole==8)+sum(hole==9)+sum(hole==10)))
print("Total Points that are Hole: " + str(total_points))
print("Largest Hole: " + str(np.max(hole)))

std_ = np.std(hole)
mean_ = np.mean(hole)

n, bins, patches = plt.hist(x=hole, bins=40, color='#0504aa', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Hole Size')
plt.ylabel('Frequency')
plt.title('Hole Size Distribution in Zebrafinch Volume')
text = "mu: " + format(mean_, '.2f') + " std: " + format(std_, '.2f')
plt.text(600000, 100000, text)
plt.yscale('log')
maxfreq = n.max()*1.3

# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.show()
