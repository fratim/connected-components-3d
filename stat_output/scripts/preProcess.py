import numpy as np
import math
import glob

hole_components = set()
points_per_component = dict()
points_per_hole_single = dict()
points_per_hole_multi = dict()


blocks_hole_components = 0
blocks_point_per_component = 0

# read in hole components (both local and global)
filenames = sorted(glob.glob("hole_components/*"))
for filename in filenames:
    blocks_hole_components += 1
    print(filename)
    hole_components_arr = np.genfromtxt(filename, delimiter=',',invalid_raise=False)
    for k in range(hole_components_arr.shape[0]):
        hole_components.add(hole_components_arr[k])

hole_components_arr = np.genfromtxt("hole_components-global.txt", delimiter=',',invalid_raise=False)
for k in range(hole_components_arr.shape[0]):
    hole_components.add(hole_components_arr[k])

# read in points per component
filenames = sorted(glob.glob("points_per_component/*"))
for filename in filenames:
    blocks_point_per_component += 1
    print(filename)
    points_per_component_arr = np.genfromtxt(filename, delimiter=',',invalid_raise=False)
    for k in range(points_per_component_arr.shape[0]):
        points_per_component[points_per_component_arr[k,0]]=points_per_component_arr[k,1]

print("Blocks Hole Components read in: " + str(blocks_hole_components))
print("Blocks Points Per Component read in: " + str(blocks_hole_components))

# create dict of points per hole
nan_components = 0
unknown_components = 0
hole_components_correct = 0

for component in hole_components:
    if math.isnan(component):
        nan_components+=1
    else:
        try:
            points_per_hole_single[(component,)] = points_per_component[component]
            hole_components_correct += 1
        except:
            unknown_components += 1
print("nan_components: " + str(nan_components))
print("unknown_components: " + str(unknown_components))
print("hole_components_correct: " + str(hole_components_correct))
print("hole_componentstotal: " + str(len(hole_components)))

#read in equaivalences
blocks_component_equivalences = 0
ids_in_multi = set()

filenames = sorted(glob.glob("component_equivalences/*"))
for filename in filenames:
    blocks_component_equivalences += 1
    component_equivalences_arr = np.genfromtxt(filename, delimiter=',',invalid_raise=False)

    if len(component_equivalences_arr.shape)==1:
        component_equivalences_arr = np.array([component_equivalences_arr[0], component_equivalences_arr[1]],ndmin=2)

    print("----------------------------------")
    print(filename + ", lines: " + str(component_equivalences_arr.shape[0]))

    for k in range(component_equivalences_arr.shape[0]):

        if component_equivalences_arr[k,0] >= 0: continue
        if component_equivalences_arr[k,1] >= 0: continue
        if component_equivalences_arr[k,0] not in hole_components: continue
        if component_equivalences_arr[k,1] not in hole_components: continue

        print("----------------------------------")
        print("k: "+ str(k) +  ",entry: " + str(component_equivalences_arr[k,:]))

        found = False
        if component_equivalences_arr[k,0] in ids_in_multi:
            if component_equivalences_arr[k,1] in ids_in_multi:
                for elem in points_per_hole_multi.keys():
                    if component_equivalences_arr[k,0] in elem and component_equivalences_arr[k,1] in elem:
                        found = True
                        print(elem)
                if found:
                    print("elements already equivalent!")
                    continue

        points_entry_0 = 0
        elem_entry_0 = ()
        points_entry_1 = 0
        elem_entry_1 = ()
        found_0 = False
        found_1 = False

        # find first component
        if component_equivalences_arr[k,0] not in ids_in_multi:
            points_entry_0 = points_per_hole_single[(component_equivalences_arr[k,0],)]
            elem_entry_0 = (component_equivalences_arr[k,0],)
            found_0 = True
            ids_in_multi.add(component_equivalences_arr[k,0])
            del points_per_hole_single[elem_entry_0]
            print("early found element 0: " + str(elem_entry_0) + ", points: " + str(points_entry_0))
        else:
            for elem in points_per_hole_multi.keys():
                if component_equivalences_arr[k,0] in elem:
                    points_entry_0 = points_per_hole_multi[elem]
                    elem_entry_0 = elem
                    found_0 = True
                    print("found element 0: " + str(elem_entry_0) + ", points: " + str(points_entry_0))
                    del points_per_hole_multi[elem_entry_0]
                    break

        # find second component
        if component_equivalences_arr[k,1] not in ids_in_multi:
            points_entry_1 = points_per_hole_single[(component_equivalences_arr[k,1],)]
            elem_entry_1 = (component_equivalences_arr[k,1],)
            found_1 = True
            ids_in_multi.add(component_equivalences_arr[k,1])
            del points_per_hole_single[elem_entry_1]
            print("early found element 1: " + str(elem_entry_1) + ", points: " + str(points_entry_1))

        else:
            for elem in points_per_hole_multi.keys():
                if component_equivalences_arr[k,1] in elem:
                    points_entry_1 = points_per_hole_multi[elem]
                    elem_entry_1 = elem
                    found_1 = True
                    print("found element 1: " + str(elem_entry_1) + ", points: " + str(points_entry_1))
                    del points_per_hole_multi[elem_entry_1]
                    break

        if found_0==False or found_1==False:
            raise ValueError("On component not found")

        new_element = elem_entry_0 + elem_entry_1
        points_added = points_entry_0 + points_entry_1
        points_per_hole_multi[new_element]=points_added

        print("new element added: " + str(new_element) + " -> " + str(points_added))

g = open("multi_holes.txt", "w+")
for entry in points_per_hole_multi.keys():
    g.write(str(int(points_per_hole_multi[entry])).zfill(25)+"\n")
g.close()

g = open("single_holes.txt", "w+")
for entry in points_per_hole_single.keys():
    g.write(str(int(points_per_hole_single[entry])).zfill(25)+"\n")
g.close()
