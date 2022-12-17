import numpy as np
import skimage.io, skimage.transform, skimage.measure
from alive_progress import alive_bar
import os


class PointCloud:
  def __init__(self, depth_map):
    self.depth_map = depth_map
    self.point_cloud = self.__create_point_clound()

  def __create_point_clound(self):
    img_loader_x = skimage.io.imread('scanner/normalized_vectors_x_plus_half.tif')
    img_loader_y = skimage.io.imread('scanner/normalized_vectors_y_plus_half.tif')
    hr_nv_x = skimage.img_as_float(img_loader_x) - 0.5
    hr_nv_y = skimage.img_as_float(img_loader_y) - 0.5

    if hr_nv_x.shape[0] != self.depth_map.shape[0]:
      factor = int(hr_nv_x.shape[0] / self.depth_map.shape[0])
      nv_x = skimage.measure.block_reduce(hr_nv_x, (factor, factor), np.mean)
      nv_y = skimage.measure.block_reduce(hr_nv_y, (factor, factor), np.mean)
    else:
      nv_x = hr_nv_x
      nv_y = hr_nv_y

    points_x = nv_x * self.depth_map
    points_y = nv_y * self.depth_map

    points_x = np.reshape(points_x, (points_x.size,))
    points_y = np.reshape(points_y, (points_y.size,))
    points_z = np.reshape(self.depth_map, (self.depth_map.size,))

    points_x = points_x[points_x != 0]
    points_y = points_y[points_y != 0] * (-1.0) # photoneo scanner y-axis convention
    points_z = points_z[points_z != 0]

    point_cloud = np.column_stack((points_x, points_y, points_z))

    return point_cloud

  def create_ply(self, file_name):
    print("Generating " + file_name + ".ply ### Size: " + str(self.point_cloud.shape[0]) + " vertices")

    if os.path.exists("ply/" + file_name + ".ply"):
      os.remove("ply/" + file_name + ".ply")

    file = open("ply/" + file_name + ".ply", "a")

    file.write("ply\n")
    file.write("format ascii 1.0\n")
    file.write("element vertex " + str(self.point_cloud.shape[0]) + "\n")
    file.write("property float32 x\n")
    file.write("property float32 y\n")
    file.write("property float32 z\n")
    file.write("end_header\n")

    with alive_bar(self.point_cloud.shape[0]) as bar:
      for i in range(0, self.point_cloud.shape[0]):
        file.write(str(self.point_cloud[i, 0]) + " " + str(self.point_cloud[i, 1]) + " " + str(self.point_cloud[i, 2]) + "\n")
        bar()

    file.close()