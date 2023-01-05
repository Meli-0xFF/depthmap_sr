import os
import torch
import cv2 as cv
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from data_preparation.normalization import Normalization
from data_preparation.filling import fill_depth_map
from metrics import get_canny_mask



class DepthMapSRDataset(Dataset):
  def __init__(self, name, train=True, train_part=0.7, task='depth_map_sr', norm=False):
    self.name = name
    self.train = train
    self.train_part = train_part
    self.task = task
    self.norm = norm

    assert os.path.isfile('dataset/' + self.name + '.npy'), "Dataset '" + self.name + "' does not exist"
    self.data = np.load('dataset/' + self.name + '.npy', allow_pickle=True).tolist()

    shuffler = np.random.permutation(len(self.data["hr"]))
    self.data["hr"] = self.data["hr"][shuffler]
    self.data["lr"] = self.data["lr"][shuffler]
    self.data["tx"] = self.data["tx"][shuffler]
    self.data["dm"] = self.data["dm"][shuffler]
    self.data["cm"] = self.data["cm"][shuffler]

    self.train_data = {"lr": self.data["lr"][:int(len(self.data['tx']) * self.train_part)],
                       "tx": self.data["tx"][:int(len(self.data['tx']) * self.train_part)],
                       "hr": self.data["hr"][:int(len(self.data['tx']) * self.train_part)],
                       "cm": self.data["cm"][:int(len(self.data['cm']) * self.train_part)],
                       "dm": self.data["dm"][:int(len(self.data['dm']) * self.train_part)]}

    self.test_data = {"lr": self.data["lr"][int(len(self.data['tx']) * self.train_part):],
                      "tx": self.data["tx"][int(len(self.data['tx']) * self.train_part):],
                      "hr": self.data["hr"][int(len(self.data['tx']) * self.train_part):],
                      "cm": self.data["cm"][int(len(self.data['cm']) * self.train_part):],
                      "dm": self.data["dm"][int(len(self.data['dm']) * self.train_part):]}

    if self.norm:
      self.normalization = Normalization(self.name)

  def __len__(self):
    if self.train:
      return len(self.train_data['tx'])
    return len(self.test_data['tx'])

  def __getitem__(self, idx):
    if self.train:
      sample = [self.train_data['lr'][idx],
                self.train_data['tx'][idx],
                self.train_data['hr'][idx],
                self.train_data['dm'][idx],
                self.train_data['cm'][idx]]
    else:
      sample = [self.test_data['lr'][idx],
                self.test_data['tx'][idx],
                self.test_data['hr'][idx],
                self.test_data['dm'][idx],
                self.test_data['cm'][idx]]

    to_tensor = transforms.Compose([transforms.ToTensor()])

    sample[0] = to_tensor(sample[0])
    sample[1] = to_tensor(sample[1])
    sample[2] = to_tensor(sample[2])
    sample[3] = to_tensor(sample[3])
    sample[4] = to_tensor(sample[4])

    if self.norm:
      sample[0] = self.normalization.normalize_sample(sample[0], "depth")
      sample[1] = self.normalization.normalize_sample(sample[1], "guide")
      sample[2] = self.normalization.normalize_sample(sample[2], "depth")

    return sample[0], sample[1], sample[2], sample[3], sample[4]


def create_dataset(name, hr_dir, lr_dir, textures_dir, scale_lr=True, fill=True, def_maps=False, canny=False):
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

  if fill:
    depth_max = 0
    for i in range(len(data["hr"])):
      m = np.max(data["hr"][i])
      if m > depth_max:
        depth_max = m

  print("===> Loading LR depth maps")
  idx = 0
  for file in sorted(os.listdir(lr_dir)):
    lr_depth_map = cv.imread(lr_dir + file, cv.IMREAD_ANYDEPTH)
    if lr_depth_maps is None:
      if scale_lr:
        lr_depth_maps = np.empty((dataset_size, hr_depth_maps.shape[1], hr_depth_maps.shape[2]), dtype=float)
      else:
        lr_depth_maps = np.empty((dataset_size, lr_depth_map.shape[0], lr_depth_map.shape[1]), dtype=float)

    if fill:
      lr_depth_map = fill_depth_map(lr_depth_map, depth_max + 1)

    if scale_lr:
      lr_tensor = torch.from_numpy(np.expand_dims(lr_depth_map, 0))
      lr_tensor = torch.nn.functional.interpolate(lr_tensor.unsqueeze(0),
                                                  size=(hr_depth_maps.shape[1], hr_depth_maps.shape[2]),
                                                  mode='bilinear',
                                                  align_corners=False)
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

  if def_maps:
    def_maps = np.empty((len(data["hr"]), data["hr"][0].shape[0], data["hr"][0].shape[1]), dtype=float)
    for i in range(len(data["hr"])):
      def_maps[i] = (data["hr"][i] > 0).astype(float)
    data['dm'] = def_maps

  if fill:
    for i in range(len(data["hr"])):
      data["hr"][i] = fill_depth_map(data["hr"][i], depth_max + 1)

  if canny:
    canny_masks = np.empty((len(data["hr"]), data["hr"][0].shape[0], data["hr"][0].shape[1]), dtype=float)
    for i in range(len(data["hr"])):
      hr_tensor = torch.from_numpy(data["hr"][i])
      hr_tensor.to(torch.device("cuda"))
      hr_tensor = hr_tensor[None, None, :]
      hr_max = torch.max(hr_tensor)
      hr_min = torch.min(hr_tensor)
      hr_tensor = hr_tensor - hr_min
      hr_tensor = hr_tensor / (hr_max - hr_min)
      canny_mask_tensor = get_canny_mask(hr_tensor).cpu()
      canny_masks[i] = canny_mask_tensor.numpy().astype(float)
    data['cm'] = canny_masks

  print("===> SAVING .npy file ...")
  np.save("dataset/" + name + ".npy", data, allow_pickle=True)

  return data