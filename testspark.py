from pyspark import SparkContext
import numpy as np

def foo(inplist):
    bz = inplist[0]
    by = inplist[1]

    return bz+by


sc = SparkContext(appName="FillBubbles")
input_data = [[0,0], [0,1], [10,0]]
distrData = sc.parallelize(input_data)
data_out = distrData.map(foo).reduce(lambda a,b:a+b)

print("data_out is: " + str(data_out))
