import numpy as np
import param
from pyspark import SparkContext
import steps_defined

sc = SparkContext(appName="FillBubbles")

input_data = [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[0,1],[1,1],[2,1],[3,1],[4,1],[5,1],[6,1],[7,1]]
distrData = sc.parallelize(input_data)

returnsum_1 = distrData.map(steps_defined.execStep1).reduce(lambda a,b:a+b)
print("returnsum 1 is:" + str(returnsum_1))
print("------------------------------------------")

returnsum_2A = distrData.map(steps_defined.execStep2A).reduce(lambda a,b:a+b)
print("returnsum 2A is:" + str(returnsum_2A))
print("------------------------------------------")


return_2B = steps_defined.execStep2B([returnsum_2A])
print("return 2B is:" + str(return_2B))
print("------------------------------------------")

returnsum_3 = distrData.map(steps_defined.execStep3).reduce(lambda a,b:a+b)
print("returnsum 3 is:" + str(returnsum_3))
print("------------------------------------------")
