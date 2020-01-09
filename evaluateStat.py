import numpy as np
import math

hole_components_arr = np.genfromtxt('stat_output/hole_components.txt', delimiter=',',invalid_raise=False)
points_per_component_arr = np.genfromtxt('stat_output/points_per_component.txt', delimiter=',',invalid_raise=False)

hole_components = set()
points_per_component = dict()
points_per_holecomponent = dict()

for k in range(hole_components_arr.shape[0]):
    hole_components.add(hole_components_arr[k])

for k in range(points_per_component_arr.shape[0]):
    points_per_component[points_per_component_arr[k,0]]=points_per_component_arr[k,1]

for component in hole_components:
    if math.isnan(component):
        print("isNan")
    else:
        try:
            points_per_holecomponent[component] = points_per_component[component]
        except:
            print("unknown component: " + str(component))
