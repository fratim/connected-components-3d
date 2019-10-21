import numpy as np
from numba import njit, types
from numba.typed import Dict
import pickle
from functions import readData, IdiToIdx

print("HERE")

z_range = np.arange(10,36)
print(z_range)

#iteration2 (4 blocks)
iteration = 2
finished = False

while finished==False:

    print("new Iteration!!")
    block_size = 2**iteration
    bz_global_range = z_range[::block_size]
    for bz_global in bz_global_range:

        print("new block")

        for bz in [bz_global, bz_global+int(block_size/2)]:
            print(bz)

    iteration = iteration + 1
    if z_range[0]+block_size >= z_range[-1]:
         finished = True
         print ("DONE")


# r = r[::2]
#
# for i in r:
#     print(i)

# counter = 0
# for i in range(20):
#  counter = counter + 1
#  if counter%5 == 0:
#      print("Count is: " + str(counter))



# labels = readData([1],"/home/frtim/wiring/raw_data/segmentations/Zebrafinch/0000")
# print(labels.shape)
#
# labels = readData([1],"/home/frtim/wiring/raw_data/segmentations/Zebrafinch/0128")
# print(labels.shape)
#
# labels = readData([1],"/home/frtim/wiring/raw_data/segmentations/Zebrafinch/2432")
# print(labels.shape)

#
# f = open("/home/frtim/Desktop/test.txt", "a+")
# f.write("1\n")
# f.close()
#
# f = open("/home/frtim/Desktop/test.txt", "a+")
# f.write("4\n")
# f.close()
#
# f = open("/home/frtim/Desktop/test.txt", "a+")
# f.write("3\n")
# f.close()
#
# f = open("/home/frtim/Desktop/test.txt", "a+")
# f.write("2\n")
# f.close()
#
# for slurm_file in *slurm
# do
#    sbatch $slurm_file;
# done
#
# python preparation.py
#
# for bz in {0..3}
# do
#     for by in {0..3}
#     do
#         for bx in {0..3}
#         do
#
#             python stepOne.py $bz $by $bx
#
#         done
#     done
# done

# @njit
# def test():
#     a = dict()
#     a[10]=11
#
#     return a
#
# b = test()
# print(b[10])

#
# a = np.zeros((1,1), dtype=np.int64)
# a = 0x7FFFFFFFFFFFFFFF
# print(a)

# a = Dict.empty(key_type=types.int64,value_type=types.int64)
# # a[10]=11
#
# b = dict()
# # b = {**b, **a}
# b[9]=10
#
# a.update(b)
#
# filename = "/home/frtim/Desktop/test.pickle"
# print(filename)
# f = open(filename, 'wb')
# pickle.dump(a, f)
#
# print(a[10])
# f.close()
#
# temp = dict()
# temp[0]=1
# print(temp[-1])

# sample_name = "ZF_concat_2to4_512_512"
#
# print(sample_name[-3:])
# print(sample_name[-7:-4])

# @njit
# def test(dict_a):
#     print(dict_a[10])
#     print(dict_a[11])
#
#
# a = Dict.empty(key_type=types.int64,value_type=types.int64)
# b = Dict.empty(key_type=types.int64,value_type=types.int64)
#
# a[10]=10
# b[11]=12
#
# a.update(b)
# # a = {**a, **b}
#
# test(a)
# a = dict()
# a[1]=2
# a[5]=3

# key_set = set(a.keys())
# # print(key_set.pop())
# # print(key_set.pop())
# key_set.add(6)
#
# for a in key_set:
#     print(a)

# @njit
# def test():
#     a = set([1,2,3])
#     if 10 in a:
#         print("HOSSA")
# test()

#
# neighbot_labels = dict()
# neighbot_labels[-10] = [4,5,6]
# print(neighbot_labels[-10])
# neighbot_labels[-10].append(100)
# print(neighbot_labels[-10])
# neighbor = [[]]
#
# print(is neighbor[0])
# try:
#   print(x)
# except:
#   print("Something went wrong")
#
# print("The 'try except' is finished")
#
#
# print(np.int64(10).itemsize)

# x = [0,1,2,3,4,-5]
# b = list(filter(lambda a: a > 0, x))
# print(b)

# a = np.ones((3,3))
# a[2,2] = -2
# a[2,1]= -3
# print(a)
#
# a[a<0] = a[a<0] - 20
# print(a)
# b=a>1
# a[b]=0
# print(a)
# labels_filled = np.ones((10,10,10))
# labels = np.zeros((10,10,10))
#
# new = np.zeros(labels.shape)
#
# print(new.shape)
#
# wholes = np.subtract(labels_filled,labels)

# print(wholes)

# print("max_label is: " + str(np.max(wholes)))
# print("min_label is: " + str(np.min(wholes)))

#
# associated_comp = {}
#
# associated_comp[1]=2
# associated_comp[5]=2
# associated_comp[8]=1
# associated_comp[9]=0
# associated_comp[2]=0
#
# print(associated_comp)
# filtered = {k: v for k, v in associated_comp.items() if v != 0}
# print(filtered)
# # import sys
#
# a = np.uint8(5)
# b = np.uint16(5)
#
# print(sizeof a)
# print(sys.getsizeof(b))

#
# import numpy as np
# x = np.linspace(1,100,100)
#
# print(x[::2])
# print(x[::3])
# print(x[10:40:6])


# arr = [np.zeros((),dtype=np.uint8) for _ in range(5)]
#
# arr[0] = np.append(arr[0],5)
# arr[1] = np.append(arr[1],6)
# arr[1] = np.append(arr[1],7)
# print(arr[0])
# print(arr[1])
# print(arr)

# adj_comp = np.ones((3,1))*-2
# test.append(1)
# print(test)
# test.append(2)

# import numpy as np
#
# a = np.array([[1, 2],
#            [3, 4]])
# b = np.array([1,1])
#
# print("a: " + str(a))
# print("b: " + str(b))
#
# print(str(tuple(b)))
# print(str(a[b]))


# a = np.repeat(2,6)
# print(a)
#
# print(np.max(np.absolute([5,0])))

# from scipy.spatial import ConvexHull, convex_hull_plot_2d
# points = np.random.rand(30, 2)   # 30 random points in 2-D
# hull = ConvexHull(points)
#
# import matplotlib.pyplot as plt
# plt.plot(points[:,0], points[:,1], 'o')
# print(points)
# print("Simplices: ")
# print(hull.simplices)
# for simplex in hull.simplices:
#     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
#
# plt.show()
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def randrange(n, vmin, vmax):
#     '''
#     Helper function to make an array of random numbers having shape (n, )
#     with each number distributed Uniform(vmin, vmax).
#     '''
#     return (vmax - vmin)*np.random.rand(n) + vmin
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# n = 100
#
# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, c=c, marker=m)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()
