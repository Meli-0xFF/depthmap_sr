#!/usr/bin/python

# command-line arguments:
#     - arg1: input depth map type [raw/out]
#     - arg2: scaling factor [integer]
#     - arg3: HR directory path
#     - arg4: LR directory path
# Example: python repair_expanded_maps.py C:/Users/lab/Downloads/david/david/pipeline_tif_textures/ ../data/david/

import os
import sys
import cv2 as cv
import numpy as np
import shutil

input_directory = sys.argv[1]
output_directory = sys.argv[2]

for filename in os.listdir(input_directory):
  file = os.path.join(input_directory, filename)
  if os.path.isfile(file):
    if filename.__contains__("out"):
      repaired_file = output_directory + "depth_map_out/" + filename
      os.makedirs(os.path.dirname(repaired_file), exist_ok=True)
      print("-> FILE " + file)

      depth_map = cv.imread(file, cv.IMREAD_ANYDEPTH)

      mask = np.zeros((depth_map.shape[0], depth_map.shape[1]), dtype=float)
      mask[::2, ::2] = 1
      mask[1::2, 1::2] = 1
      repaired_depth_map = np.reshape(depth_map[mask > 0], (depth_map.shape[0], int(depth_map.shape[1] / 2)))

      cv.imwrite(repaired_file, repaired_depth_map)

  if filename.__contains__("laser"):
    dst = output_directory + "texture_laser/" + filename
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print("-> FILE " + file)
    shutil.copyfile(file, dst)
