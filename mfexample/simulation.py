#!/usr/bin/env python

import datetime
import math
import os
import sys
import time

if(len(sys.argv) < 2 ):
	sys.stdout.write("usage: simulation.py <integer>\n")
	sys.exit(1)

sys.stdout.write(datetime.datetime.now().strftime('Starting %Y %d %b %H:%M:%S\n'))
sys.stdout.write("Please enter some text: ");

inp_file = sys.stdin.readline()
inp_file2 = sys.stdin.readline()

print(type(inp_file))
print(inp_file)
print(len(inp_file))

print(type(inp_file2))
print(inp_file2)
print(len(inp_file2))

if (int(inp_file)!=1):
  print("aborting, input is: " + str(inp_file))

else:

  sys.stdout.write('The input contains: %s\n' % sys.stdin.readline() )
  os.system("echo Running on host `hostname --fqdn`");

  x = float(sys.argv[1])
  for i in range(0, 5):
      time.sleep(1)
      y = math.sqrt(x)
      sys.stdout.write("x = "+str(y)+"\n")
      x = y

  sys.stdout.write(datetime.datetime.now().strftime('Finished %Y %d %b %H:%M:%S\n'))
