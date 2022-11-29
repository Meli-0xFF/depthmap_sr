import os
import torch
import numpy as np
from math import sqrt
from dataset import *
from torch.utils.data import DataLoader


class Normalization:
  def __init__(self, dataset_name):
    self.norm_file_path = 'dataset/' + dataset_name + '_norm.npy'

    assert os.path.isfile(self.norm_file_path), "Normalization file for dataset '" + dataset_name + "' does not exist"
    if os.path.isfile(self.norm_file_path):
      self.norm_data = np.load(self.norm_file_path, allow_pickle=True).tolist()
      self.guide_interval = self.norm_data["guide"]
      self.depth_interval = self.norm_data["depth"]


  def normalize_sample(self, img, img_type):
    if img_type == "depth":
      img = torch.where(img == 0.0, -1.0, img)

    img = torch.where(img >= 0, img - self.norm_data[img_type][0], img)
    img = torch.where(img >= 0, img / (self.norm_data[img_type][1] - self.norm_data[img_type][0]), img)

    return img

  def recover_normalized_sample(self, img, img_type):
    img = torch.where(img >= 0, img * (self.norm_data[img_type][1] - self.norm_data[img_type][0]), img)
    img = torch.where(img >= 0, img + self.norm_data[img_type][0], img)

    if img_type == "depth":
      img = torch.where(img < 0, 0, img)

    return img
