import cv2 as cv
from utils import *
from pointcloud import *
import matplotlib.pyplot as plt

def main():
  hr_depth_map = cv.imread("../data/led-warior/depth_map_out_1219.tif", cv.IMREAD_ANYDEPTH)
  create_ply(hr_depth_map, "hr_warior")

  ds = Downsampling(hr_depth_map, 2)
  lr_depth_map = ds.median_method()
  create_ply(lr_depth_map, "lr_warior")

  plt.imshow(lr_depth_map, cmap="winter")
  plt.show()

if __name__ == "__main__":
  main()