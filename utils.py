import numpy as np
from constants import *

class Downsampling:
  def __init__(self, hr_depth_map, factor):
    self.hr_depth_map = hr_depth_map
    self.factor = factor

    self.lr_depth_map = np.zeros((int(self.hr_depth_map.shape[0] / self.factor), int(self.hr_depth_map.shape[1] / self.factor)))

  def medianMethod(self):
    edge_treshold = 0.1

    for i in range(0, int(HR_ROWS / self.factor) + (1 if HR_ROWS % self.factor != 0 else 0)):
      for j in range(0, int(HR_COLUMNS / self.factor) + (1 if HR_COLUMNS % self.factor != 0 else 0)):

        w_heigth = self.factor if (i * self.factor) < HR_ROWS else HR_ROWS % self.factor
        w_width = self.factor if (j * self.factor) < HR_COLUMNS else HR_COLUMNS % self.factor

        w = self.hr_depth_map[i * self.factor:i * self.factor + w_heigth, j * self.factor:j * self.factor + w_width].flatten()

        w_avg = np.mean(w)
        w_min = np.min(w)
        w_max = np.max(w)

        fg = w <= w_avg

        if (w_max - w_min) > edge_treshold:
          self.lr_depth_map[i, j] = np.median(w)
        else:
          self.lr_depth_map[i, j] = np.median(np.ma.masked_array(w, mask=fg))

    return self.lr_depth_map