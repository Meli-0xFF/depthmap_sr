import cv2 as cv
from utils import *


def main():
  hr_depth_map = cv.imread("../data/led-warior/depth_map_out_1219.tif", cv.IMREAD_ANYDEPTH)

  ds = Downsampling(hr_depth_map, 2)
  lr_depth_map = ds.medianMethod()

  print("HR: " + str(hr_depth_map.shape))
  print("LR: " + str(lr_depth_map.shape))

  cv.imshow("hr depth map", hr_depth_map)
  cv.imshow("lr depth map", lr_depth_map)

  cv.waitKey(0)
  cv.destroyAllwindows()


if __name__ == "__main__":
  main()