import numpy as np
from alive_progress import alive_bar

class Downsampling:
  def __init__(self, hr_depth_map, factor):
    self.hr_depth_map = hr_depth_map
    self.factor = factor

    self.lr_depth_map = np.zeros((int(self.hr_depth_map.shape[0] / self.factor), int(self.hr_depth_map.shape[1] / self.factor)))

  def median_method(self):
    edge_treshold = 0.1

    lr_rows = int(self.hr_depth_map.shape[0] / self.factor) + (1 if self.hr_depth_map.shape[0] % self.factor != 0 else 0)
    lr_columns = int(self.hr_depth_map.shape[1] / self.factor) + (1 if self.hr_depth_map.shape[1] % self.factor != 0 else 0)

    print("# downsampling HR depth map")

    with alive_bar(lr_rows * lr_columns) as bar:
      for i in range(0, lr_rows):
        for j in range(0, lr_columns):

          w_heigth = self.factor if (i * self.factor) < self.hr_depth_map.shape[0] else self.hr_depth_map.shape[0] % self.factor
          w_width = self.factor if (j * self.factor) < self.hr_depth_map.shape[1] else self.hr_depth_map.shape[1] % self.factor

          w = self.hr_depth_map[i * self.factor:i * self.factor + w_heigth, j * self.factor:j * self.factor + w_width].flatten()

          w_avg = np.mean(w)
          w_min = np.min(w)
          w_max = np.max(w)

          fg = w[w <= w_avg]

          if (w_max - w_min) > edge_treshold:
            w_median = self.custom_median(w)
            if w_median == 0:
              self.lr_depth_map[i, j] = w_max
            else:
              self.lr_depth_map[i, j] = w_median
          else:
            fg_median = self.custom_median(fg)
            if fg_median == 0:
              self.lr_depth_map[i, j] = w_max
            else:
              self.lr_depth_map[i, j] = np.median(fg)
          bar()

    return self.lr_depth_map

  def custom_median(self, array):
    sorted = np.sort(array)
    return sorted[int(sorted.shape[0]/2)]