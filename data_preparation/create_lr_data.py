#!/usr/bin/python

# command-line arguments:
#     - arg1: input depth map type [raw/out]
#     - arg2: scaling factor [integer]
#     - arg3: HR directory path
#     - arg4: LR directory path
# Example: python create_lr_data.py out 2 ../data/led-warior/ ../data/lr-led-warior/

from downsample import *
import os
import sys
import cv2 as cv
import cupy as cp

f_type = sys.argv[1]
ds_factor = sys.argv[2]
hr_directory = sys.argv[3]
lr_directory = sys.argv[4]

i = 1
file_list = os.listdir(hr_directory)

for hr_filename in file_list:
  hr_file = os.path.join(hr_directory, hr_filename)
  if os.path.isfile(hr_file):
    if hr_filename.__contains__(f_type):
      lr_file = lr_directory + "lr_" + hr_filename
      os.makedirs(os.path.dirname(lr_file), exist_ok=True)
      print("-> FILE " + hr_file + " " + str(i) + "/" + str(len(file_list)))

      hr_depth_map = cv.imread(hr_file, cv.IMREAD_ANYDEPTH)

      ds = Downsampling(hr_depth_map, int(ds_factor))
      lr_depth_map = ds.median_method()

      cv.imwrite(lr_file, cp.asnumpy(lr_depth_map))
      i = i + 1