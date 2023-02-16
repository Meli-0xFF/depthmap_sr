import numpy as np
import skimage.io, skimage.transform, skimage.measure
from alive_progress import alive_bar
import os


class NyuPointCloud:
  def __init__(self, depth_map):
    self.depth_map = depth_map
    self.point_cloud = self.__create_point_clound()

  def __create_point_clound(self):
    fx = 5.8262448167737955e+02
    fy = 5.8269103270988637e+02
    cx = 3.1304475870804731e+02
    cy = 2.3844389626620386e+02
    depth_scale = 1.0

    idx_x = np.arange(self.depth_map.shape[1])
    idx_y = np.arange(self.depth_map.shape[0])
    u = np.vstack([idx_x]*self.depth_map.shape[0])
    v = np.column_stack([idx_y]*self.depth_map.shape[1])

    z = self.depth_map / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_x = np.reshape(x, (x.size,))
    points_y = np.reshape(y, (y.size,))
    points_z = np.reshape(z, (self.depth_map.size,))

    points_x = points_x[points_x != 0]
    points_y = points_y[points_y != 0]
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
