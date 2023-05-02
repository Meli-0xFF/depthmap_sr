import torch
import numpy as np
from alive_progress import alive_bar
from skimage import measure, segmentation
import skimage.io, skimage.transform
import matplotlib.pyplot as plt
import matplotlib as mpl
import cupy as cp

def get_plane(img):
  img_loader_x = skimage.io.imread('scanner/normalized_vectors_x_plus_half.tif')
  img_loader_y = skimage.io.imread('scanner/normalized_vectors_y_plus_half.tif')
  nv_x = cp.asarray(skimage.img_as_float(img_loader_x) - 0.5)
  nv_y = cp.asarray(skimage.img_as_float(img_loader_y) - 0.5)

  img_mean = cp.mean(img[img != 0])
  mean_error_map = cp.power((img - img_mean), 2)
  mean_error_map = cp.where(img > 0, mean_error_map, cp.infty)

  mean_sorted = cp.dstack(cp.unravel_index(cp.argsort(mean_error_map.ravel()), img.shape))[0]
  depth_sorted = cp.dstack(cp.unravel_index(cp.argsort(img.ravel()), img.shape))[0]
  a = mean_sorted[:20000]
  b = depth_sorted
  acceptable_mean = mean_error_map[a[19999, 0], a[19999, 1]]
  acceptable_depth = img[b[440000, 0], b[440000, 1]]

  img1 = cp.where(img == 0, 1.0, img)
  x = nv_x * img1
  y = nv_y * img1 * (-1.0)
  z = img1.copy()

  mesh_points = cp.empty((int(img.shape[0] / 20) * int(img.shape[1] / 20), 2))
  for i in range(int(img.shape[0] / 20)):
    for j in range(int(img.shape[1] / 20)):
      mesh_points[int(i * img.shape[1] / 20) + j, 0] = i * 20
      mesh_points[int(i * img.shape[1] / 20) + j, 1] = j * 20

  mean_points = None
  depth_points = None
  for i in range(mesh_points.shape[0]):
    if mean_error_map[int(mesh_points[i, 0]), int(mesh_points[i, 1])] < acceptable_mean:
      if mean_points is None:
        mean_points = cp.array([mesh_points[i]])
      else:
        mean_points = cp.vstack([mean_points, mesh_points[i]])

    if img[int(mesh_points[i, 0]), int(mesh_points[i, 1])] > acceptable_depth:
      if depth_points is None:
        depth_points = cp.array([mesh_points[i]])
      else:
        depth_points = cp.vstack([depth_points, mesh_points[i]])

  mean_points = mean_points[mean_points[:, 1].argsort()]
  depth_points = depth_points[depth_points[:, 1].argsort()]

  '''
  cmap = mpl.cm.get_cmap("winter").copy()
  cmap.set_under(color='black')
  plt.figure(plt.figure('HR Depth map'))
  plt.imshow(img.get(), cmap=cmap, vmin=0.001)
  
  for i in range(mean_points.shape[0]):
    plt.scatter(int(mean_points[i, 1]), int(mean_points[i, 0]), c='pink', marker='o')

  for i in range(depth_points.shape[0]):
    plt.scatter(int(depth_points[i, 1]), int(depth_points[i, 0]), c='yellow', marker='o')

  plt.show()
  '''

  mean_X = cp.empty((mean_points.shape[0]))
  mean_Y = cp.empty((mean_points.shape[0]))
  mean_Z = cp.empty((mean_points.shape[0]))

  for i in range(mean_points.shape[0]):
    mean_X[i] = x[int(mean_points[i, 0]), int(mean_points[i, 1])]
    mean_Y[i] = y[int(mean_points[i, 0]), int(mean_points[i, 1])]
    mean_Z[i] = z[int(mean_points[i, 0]), int(mean_points[i, 1])]

  real_mean_points = cp.c_[mean_X, mean_Y, mean_Z]

  depth_X = cp.empty((depth_points.shape[0]))
  depth_Y = cp.empty((depth_points.shape[0]))
  depth_Z = cp.empty((depth_points.shape[0]))

  for i in range(depth_points.shape[0]):
    depth_X[i] = x[int(depth_points[i, 0]), int(depth_points[i, 1])]
    depth_Y[i] = y[int(depth_points[i, 0]), int(depth_points[i, 1])]
    depth_Z[i] = z[int(depth_points[i, 0]), int(depth_points[i, 1])]

  real_depth_points = cp.c_[depth_X, depth_Y, depth_Z]

  A = real_mean_points[0]
  B = real_depth_points[int(real_depth_points.shape[0] / 2)]
  C = real_mean_points[real_mean_points.shape[0] - 1]
  v1 = C - A
  v2 = B - A

  '''
  cmap = mpl.cm.get_cmap("winter").copy()
  cmap.set_under(color='black')
  plt.figure(plt.figure('HR Depth map'))
  plt.imshow(img, cmap=cmap, vmin=0.001)
  plt.scatter(int(mean_points[0, 1]), int(mean_points[0, 0]), c='red', marker='o')
  plt.scatter(int(depth_points[int(depth_points.shape[0] / 2), 1]), int(depth_points[int(depth_points.shape[0] / 2), 0]), c='red', marker='o')
  plt.scatter(int(mean_points[mean_points.shape[0] - 1, 1]), int(mean_points[mean_points.shape[0] - 1, 0]), c='red', marker='o')
  plt.show()
  '''

  normal = cp.cross(v1, v2)

  d = -cp.sum(normal * A)
  full_plane = -d / (normal[0] * nv_x + normal[1] * nv_y * (-1.0) + normal[2])
  img_dis = abs(normal[0] * x + + normal[1] * y + normal[2] * img + d) / cp.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
  img_dis = cp.where(img == 0, 0, img_dis)

  return full_plane, img_dis


def fill_depth_map(img, background_value):
  mask = np.where(img > 0, 0, 1).astype(np.uint8)

  labels, holes = measure.label(mask, background=0, return_num=True)
  img = cp.asarray(img)
  labels = cp.asarray(labels)
  print("Filling " + str(holes) + " holes")
  with alive_bar(holes) as bar:
    for i in range(1, holes + 1):
      label = labels == i

      label_img_border_pixels = cp.sum(label.astype(cp.int32)[0, :]) + \
                                cp.sum(label.astype(cp.int32)[label.shape[0] - 1, :]) + \
                                cp.sum(label.astype(cp.int32)[1:(label.shape[1] - 1), 0]) + \
                                cp.sum(label.astype(cp.int32)[1:(label.shape[1] - 1), label.shape[1] - 1])

      if label_img_border_pixels == 0:
        top = cp.roll(label, 1, axis=0)
        bottom = cp.roll(label, -1, axis=0)
        right = cp.roll(label, 1, axis=1)
        left = cp.roll(label, -1, axis=1)
        hole_border = cp.logical_or(cp.logical_or(top, bottom), cp.logical_or(right, left))
        border_values = img * hole_border

      background = label_img_border_pixels > 0

      if not background:
        line_values = border_values.max(1)
        hole = (labels == i).astype(float) * line_values[:, cp.newaxis]
      else:
        hole = (labels == i).astype(float) * 0

      bar()

      img = cp.where(hole > 0, hole, img)

  plane, dis = get_plane(img)
  object_map = cp.where(dis < 10, 0, 1.0)
  img = cp.where(img == 0, background_value, img)

  '''
  cmap = mpl.cm.get_cmap("winter").copy()
  cmap.set_under(color='black')
  plt.figure(plt.figure('HR Plane'))
  plt.imshow(object_map, cmap='gray')
  plt.figure(plt.figure('HR Depth map'))
  plt.imshow(img, cmap=cmap, vmin=0.001)
  plt.show()
  '''

  return img.get(), object_map.get()

def fill_texture(tex, def_map):
  mask = np.where(def_map > 0, 0, 1).astype(np.uint8)

  labels, holes = measure.label(mask, background=0, return_num=True)
  tex = cp.asarray(tex)
  labels = cp.asarray(labels)
  print("Filling " + str(holes) + " holes")
  with alive_bar(holes) as bar:
    for i in range(1, holes + 1):
      label = labels == i

      label_img_border_pixels = cp.sum(label.astype(cp.int32)[0, :]) + \
                                cp.sum(label.astype(cp.int32)[label.shape[0] - 1, :]) + \
                                cp.sum(label.astype(cp.int32)[1:(label.shape[1] - 1), 0]) + \
                                cp.sum(label.astype(cp.int32)[1:(label.shape[1] - 1), label.shape[1] - 1])

      if label_img_border_pixels == 0:
        top = cp.roll(label, 1, axis=0)
        bottom = cp.roll(label, -1, axis=0)
        right = cp.roll(label, 1, axis=1)
        left = cp.roll(label, -1, axis=1)
        hole_border = cp.logical_or(cp.logical_or(top, bottom), cp.logical_or(right, left))
        border_values = tex * hole_border

      background = label_img_border_pixels > 0

      if not background:
        line_values = border_values.max(1)
        hole = (labels == i).astype(float) * line_values[:, cp.newaxis]
      else:
        hole = (labels == i).astype(float) * 0

      bar()

      tex = cp.where(hole > 0, hole, tex)
    return tex.get()
