import torch
import numpy as np
from alive_progress import alive_bar
from skimage import measure, segmentation
from evaluation.pointcloud import PointCloud


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
        hole = (labels == i).astype(float) * background_value

      bar()

      img = np.where(hole > 0, hole, img)

    return img