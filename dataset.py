import numpy as np
import os
import cv2 as cv
import torch
from torch.utils.data import Dataset


class DepthMapSRDataset(Dataset):

  def __init__(self, name, lr_transform=None, tx_transform=None, hr_transform=None, train=True, train_part=0.7):
    self.name = name
    self.lr_transform = lr_transform
    self.tx_transform = tx_transform
    self.hr_transform = hr_transform
    self.train = train
    self.train_part = train_part

    assert os.path.isfile('dataset/' + self.name + '.npy'), "Dataset '" + self.name + "' does not exist"
    self.data = np.load('dataset/' + self.name + '.npy', allow_pickle=True).tolist()

    shuffler = np.random.permutation(len(self.data["hr"]))
    self.data["hr"] = self.data["hr"][shuffler]
    self.data["lr"] = self.data["lr"][shuffler]
    self.data["tx"] = self.data["tx"][shuffler]

    self.train_data = {"lr" : self.data["lr"][:int(len(self.data['tx']) * self.train_part)],
                       "tx" : self.data["tx"][:int(len(self.data['tx']) * self.train_part)],
                       "hr": self.data["hr"][:int(len(self.data['tx']) * self.train_part)]}

    self.test_data = {"lr": self.data["lr"][int(len(self.data['tx']) * self.train_part):],
                      "tx": self.data["tx"][int(len(self.data['tx']) * self.train_part):],
                      "hr": self.data["hr"][int(len(self.data['tx']) * self.train_part):]}

  def __len__(self):
    if self.train:
      return len(self.train_data['tx'])
    return len(self.test_data['tx'])


  def __getitem__(self, idx):
    if self.train:
      sample = [self.train_data['lr'][idx], self.train_data['tx'][idx], self.train_data['hr'][idx]]
    else:
      sample = [self.test_data['lr'][idx], self.test_data['tx'][idx], self.test_data['hr'][idx]]

    if self.lr_transform:
      sample[0] = self.lr_transform(sample[0])

    if self.tx_transform:
      sample[1] = self.tx_transform(sample[1])

    if self.hr_transform:
      sample[2] = self.hr_transform(sample[2])

    return sample[0], sample[1], sample[2]

def create_dataset(name, hr_dir, lr_dir, textures_dir, scale_lr=False):
  print("--- CREATE DATASET: " + name + " ---")
  data = dict()
  hr_depth_maps = None
  lr_depth_maps = None
  textures = None
  dataset_size = len(os.listdir(hr_dir))

  print("===> Loading HR depth maps")
  idx = 0
  for file in sorted(os.listdir(hr_dir)):
    hr_depth_map = cv.imread(hr_dir + file, cv.IMREAD_ANYDEPTH)
    if hr_depth_maps is None:
      hr_depth_maps = np.empty((dataset_size, hr_depth_map.shape[0], hr_depth_map.shape[1]), dtype=float)
    hr_depth_maps[idx] = hr_depth_map
    idx += 1
  data['hr'] = hr_depth_maps

  print("===> Loading LR depth maps")
  idx = 0
  for file in sorted(os.listdir(lr_dir)):
    lr_depth_map = cv.imread(lr_dir + file, cv.IMREAD_ANYDEPTH)
    if lr_depth_maps is None:
      if scale_lr:
        lr_depth_maps = np.empty((dataset_size, hr_depth_maps.shape[1], hr_depth_maps.shape[2]), dtype=float)
      else:
        lr_depth_maps = np.empty((dataset_size, lr_depth_map.shape[0], lr_depth_map.shape[1]), dtype=float)

    if scale_lr:
      lr_tensor = torch.from_numpy(np.expand_dims(lr_depth_map, 0))
      lr_tensor = torch.nn.functional.interpolate(lr_tensor.unsqueeze(0), size=(hr_depth_maps.shape[1], hr_depth_maps.shape[2]), mode='bilinear', align_corners=False)
      lr_depth_map = lr_tensor.numpy()

    lr_depth_maps[idx] = lr_depth_map
    idx += 1
  data['lr'] = lr_depth_maps

  print("===> Loading HR textures")
  idx = 0
  for file in sorted(os.listdir(textures_dir)):
    texture = cv.imread(textures_dir + file, cv.IMREAD_ANYDEPTH)
    if textures is None:
      textures = np.empty((dataset_size, texture.shape[0], texture.shape[1]), dtype=float)
    textures[idx] = texture
    idx += 1
  data['tx'] = textures

  print("===> SAVING .npy file ...")
  np.save("dataset/" + name + ".npy", data, allow_pickle=True)

  return data


def compute_mean_and_std(name):
  lr_sum, lr_squared_sum = 0, 0
  tx_sum, tx_squared_sum = 0, 0
  hr_sum, hr_squared_sum = 0, 0

  assert os.path.isfile('dataset/' + name + '.npy'), "Dataset '" + name + "' does not exist"

  print("===> Computing dataset mean and std")
  data = np.load('dataset/' + name + '.npy', allow_pickle=True).tolist()

  for idx in range(len(data)):
    lr_sum += np.mean(data['lr'][idx])
    lr_squared_sum += np.mean(data['lr'][idx] ** 2)

    tx_sum += np.mean(data['tx'][idx])
    tx_squared_sum += np.mean(data['tx'][idx] ** 2)

    hr_sum += np.mean(data['hr'][idx])
    hr_squared_sum += np.mean(data['hr'][idx] ** 2)

  lr_mean = lr_sum / len(data)
  lr_std = (lr_squared_sum / len(data) - lr_mean ** 2) ** 0.5

  tx_mean = tx_sum / len(data)
  tx_std = (tx_squared_sum / len(data) - tx_mean ** 2) ** 0.5

  hr_mean = hr_sum / len(data)
  hr_std = (hr_squared_sum / len(data) - hr_mean ** 2) ** 0.5

  n_file = open("dataset/" + name + "_normalization.txt", "a")

  n_file.write("lr_mean, lr_std, tx_mean, tx_std, hr_mean, hr_std =" +
                             str(lr_mean) + ", " + str(lr_std) + ", " +
                             str(tx_mean) + ", " + str(tx_std) + ", " +
                             str(hr_mean) + ", " + str(hr_std) + "\n")
  n_file.close()


def get_mean_and_std(name):
  assert os.path.isfile('dataset/' + name + '_normalization.txt'), "Normalization for dataset '" + name + "' not generated"
  n_file = open("dataset/" + name + "_normalization.txt", "r")
  return [float(item) for item in n_file.readline().split('=', 1)[1].split(", ", -1)]