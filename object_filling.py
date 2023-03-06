import torch
import numpy as np
from alive_progress import alive_bar
from skimage import measure, segmentation
import skimage.io, skimage.transform
import matplotlib.pyplot as plt
import matplotlib as mpl


def get_plane(img):
  img_loader_x = skimage.io.imread('scanner/normalized_vectors_x_plus_half.tif')
  img_loader_y = skimage.io.imread('scanner/normalized_vectors_y_plus_half.tif')
  nv_x = skimage.img_as_float(img_loader_x) - 0.5
  nv_y = skimage.img_as_float(img_loader_y) - 0.5

  img_mean = np.mean(img[img != 0])
  mean_error_map = np.power((img - img_mean), 2)
  mean_error_map = np.where(img > 0, mean_error_map, np.infty)

  mean_sorted = np.dstack(np.unravel_index(np.argsort(mean_error_map.ravel()), img.shape))[0]
  depth_sorted = np.dstack(np.unravel_index(np.argsort(img.ravel()), img.shape))[0]
  a = mean_sorted[:20000]
  b = depth_sorted
  acceptable_mean = mean_error_map[a[19999, 0], a[19999, 1]]
  acceptable_depth = img[b[440000, 0], b[440000, 1]]

  img1 = np.where(img == 0, 1.0, img)
  x = nv_x * img1
  y = nv_y * img1 * (-1.0)
  z = img1.copy()

  mesh_points = np.empty((int(img.shape[0] / 20) * int(img.shape[1] / 20), 2))
  for i in range(int(img.shape[0] / 20)):
    for j in range(int(img.shape[1] / 20)):
      mesh_points[int(i * img.shape[1] / 20) + j, 0] = i * 20
      mesh_points[int(i * img.shape[1] / 20) + j, 1] = j * 20

  mean_points = None
  depth_points = None
  for i in range(mesh_points.shape[0]):
    if mean_error_map[int(mesh_points[i, 0]), int(mesh_points[i, 1])] < acceptable_mean:
      if mean_points is None:
        mean_points = np.array([mesh_points[i]])
      else:
        mean_points = np.vstack([mean_points, mesh_points[i]])

    if img[int(mesh_points[i, 0]), int(mesh_points[i, 1])] > acceptable_depth:
      if depth_points is None:
        depth_points = np.array([mesh_points[i]])
      else:
        depth_points = np.vstack([depth_points, mesh_points[i]])

  mean_points = mean_points[mean_points[:, 1].argsort()]
  depth_points = depth_points[depth_points[:, 1].argsort()]

  '''
  cmap = mpl.cm.get_cmap("winter").copy()
  cmap.set_under(color='black')
  plt.figure(plt.figure('HR Depth map'))
  plt.imshow(img, cmap=cmap, vmin=0.001)
  
  for i in range(mean_points.shape[0]):
    plt.scatter(int(mean_points[i, 1]), int(mean_points[i, 0]), c='red', marker='o')

  for i in range(depth_points.shape[0]):
    plt.scatter(int(depth_points[i, 1]), int(depth_points[i, 0]), c='yellow', marker='o')

  plt.show()
  '''

  mean_X = np.empty((mean_points.shape[0]))
  mean_Y = np.empty((mean_points.shape[0]))
  mean_Z = np.empty((mean_points.shape[0]))

  for i in range(mean_points.shape[0]):
    mean_X[i] = x[int(mean_points[i, 0]), int(mean_points[i, 1])]
    mean_Y[i] = y[int(mean_points[i, 0]), int(mean_points[i, 1])]
    mean_Z[i] = z[int(mean_points[i, 0]), int(mean_points[i, 1])]

  real_mean_points = np.c_[mean_X, mean_Y, mean_Z]

  depth_X = np.empty((depth_points.shape[0]))
  depth_Y = np.empty((depth_points.shape[0]))
  depth_Z = np.empty((depth_points.shape[0]))

  for i in range(depth_points.shape[0]):
    depth_X[i] = x[int(depth_points[i, 0]), int(depth_points[i, 1])]
    depth_Y[i] = y[int(depth_points[i, 0]), int(depth_points[i, 1])]
    depth_Z[i] = z[int(depth_points[i, 0]), int(depth_points[i, 1])]

  real_depth_points = np.c_[depth_X, depth_Y, depth_Z]

  A = real_mean_points[0]
  B = real_depth_points[0]
  C = real_mean_points[real_mean_points.shape[0] - 1]
  v1 = C - A
  v2 = B - A

  '''
  cmap = mpl.cm.get_cmap("winter").copy()
  cmap.set_under(color='black')
  plt.figure(plt.figure('HR Depth map'))
  plt.imshow(img, cmap=cmap, vmin=0.001)
  plt.scatter(int(mean_points[0, 1]), int(mean_points[0, 0]), c='red', marker='o')
  plt.scatter(int(depth_points[0, 1]), int(depth_points[0, 0]), c='red', marker='o')
  plt.scatter(int(mean_points[mean_points.shape[0] - 1, 1]), int(mean_points[mean_points.shape[0] - 1, 0]), c='red', marker='o')
  plt.show()
  '''

  normal = np.cross(v1, v2)

  d = -np.sum(normal * A)
  #plane = -(normal[0] * x + normal[1] * y + d) / normal[2]
  full_plane = -d / (normal[0] * nv_x + normal[1] * nv_y * (-1.0) + normal[2])
  img_dis = abs(normal[0] * x + + normal[1] * y + normal[2] * img + d) / np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
  img_dis = np.where(img == 0, 0, img_dis)

  return full_plane, img_dis


def fill_depth_map(img, background_value):
  mask = np.where(img > 0, 0, 1).astype(np.uint8)
  uint_mask = np.where(img > 0, 0, 255).astype(np.uint8)
  rgb_mask = np.dstack((uint_mask, uint_mask, uint_mask))

  labels, holes = measure.label(mask, background=0, return_num=True)

  print("Filling " + str(holes) + " holes")
  with alive_bar(holes) as bar:
    for i in range(1, holes + 1):
      label = labels == i
      hole_border = segmentation.mark_boundaries(np.zeros(rgb_mask.shape),
                                                 label.astype(np.int32),
                                                 (1, 0, 0),
                                                 None,
                                                 'outer')[:, :, 0].astype(float)

      #hole_border1 = np.logical_or((np.cumsum(label.astype(np.int32), axis=0) == 1), (np.cumsum(label.astype(np.int32), axis=1) == 1))
      #hole_border2 = np.logical_or(np.flipud(np.flipud(label.astype(np.int32).cumsum(axis=1)) == 1), np.fliplr(np.fliplr(label.astype(np.int32).cumsum(axis=0)) == 1))
      #hole_border = np.logical_or(hole_border1, hole_border2).astype(float)

      border_values = img * hole_border
      label_img_border_pixels = np.sum(label.astype(np.int32)[0, :]) + \
                                np.sum(label.astype(np.int32)[label.shape[0] - 1, :]) + \
                                np.sum(label.astype(np.int32)[1:(label.shape[1] - 1), 0]) + \
                                np.sum(label.astype(np.int32)[1:(label.shape[1] - 1), label.shape[1] - 1])

      background = label_img_border_pixels > 0

      if not background:
        mx = np.ma.masked_array(border_values, mask=border_values == 0)
        line_values = mx.max(1)
        hole = (labels == i).astype(float) * line_values[:, np.newaxis]
      else:

        hole = (labels == i).astype(float) * 0

      bar()

      img = np.where(hole > 0, hole, img)

  plane, dis = get_plane(img)
  object_map = np.where(dis < 10, 0, 1.0)
  img = np.where(img == 0, background_value, img)

  '''
  cmap = mpl.cm.get_cmap("winter").copy()
  cmap.set_under(color='black')
  plt.figure(plt.figure('HR Plane'))
  plt.imshow(object_map, cmap='gray')
  plt.figure(plt.figure('HR Depth map'))
  plt.imshow(img, cmap=cmap, vmin=0.001)
  plt.show()
  '''

  return img, object_map

def fill_hr_texture(tex, def_map):
  mask = np.where(def_map > 0, 0, 1).astype(np.uint8)
  uint_mask = np.where(def_map > 0, 0, 255).astype(np.uint8)
  rgb_mask = np.dstack((uint_mask, uint_mask, uint_mask))

  labels, holes = measure.label(mask, background=0, return_num=True)

  print("Filling " + str(holes) + " holes")
  with alive_bar(holes) as bar:
    for i in range(1, holes + 1):
      label = labels == i

      hole_border = segmentation.mark_boundaries(np.zeros(rgb_mask.shape),
                                                 label.astype(np.int32),
                                                 (1, 0, 0),
                                                 None,
                                                 'outer')[:, :, 0].astype(float)
      border_values = tex * hole_border
      label_img_border_pixels = np.sum(label.astype(np.int32)[0, :]) + \
                                np.sum(label.astype(np.int32)[label.shape[0] - 1, :]) + \
                                np.sum(label.astype(np.int32)[1:(label.shape[1] - 1), 0]) + \
                                np.sum(label.astype(np.int32)[1:(label.shape[1] - 1), label.shape[1] - 1])

      background = label_img_border_pixels > 0 or np.sum((labels == i).astype(float)) > 1000

      if not background:
        mx = np.ma.masked_array(border_values, mask=border_values == 0)
        line_values = mx.max(1)
        hole = (labels == i).astype(float) * line_values[:, np.newaxis]
      else:
        hole = (labels == i).astype(float) * 1

      bar()

      tex = np.where(hole > 0, hole, tex)
    return tex
