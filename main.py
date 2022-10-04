import cv2 as cv
from downsample import *
from pointcloud import *
import matplotlib.pyplot as plt
import matplotlib as mpl

def main():
  hr_depth_map = cv.imread("../data/led-warior/depth_map_out/depth_map_out_1219.tif", cv.IMREAD_ANYDEPTH)
  hr_point_coud = Pointcloud(hr_depth_map)
  hr_point_coud.create_ply("hr_warior")

  ds_factor = 4
  ds = Downsampling(hr_depth_map, ds_factor)
  lr_depth_map = ds.median_method()

  lr_point_coud = Pointcloud(lr_depth_map)
  lr_point_coud.create_ply(str(ds_factor) + "-lr_warior")

  cmap = mpl.cm.get_cmap("winter").copy()
  cmap.set_under(color='black')
  plt.imshow(lr_depth_map, cmap=cmap, vmin=0.0000001)
  plt.show()

if __name__ == "__main__":
  main()