import h5py
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib as mpl
from nyu_point_cloud import *
from data_preparation.create_normalization import create_dataset_norm_data

from data_preparation.filling import fill_depth_map
from metrics import get_canny_mask

import os
import torch
import cv2 as cv
import numpy as np

'''
f = h5py.File('data/nyu_depth_v2_labeled.mat')
print(list(f.keys()))
depth_maps = f['depths']
raw_depth_maps = f['rawDepths']
rgb_textures = f['images']

rgb_textures = np.transpose(rgb_textures, (0, 3, 2, 1))
hr_depth_maps = np.transpose(depth_maps, (0, 2, 1))
raw_depth_maps = np.transpose(raw_depth_maps, (0, 2, 1))

textures = np.empty((len(rgb_textures), hr_depth_maps.shape[1], hr_depth_maps.shape[2]), dtype=float)
lr_depth_maps_not_scaled = np.empty((len(rgb_textures), int(hr_depth_maps.shape[1]/2), int(hr_depth_maps.shape[2]/2)), dtype=float)
lr_depth_maps = np.empty((len(hr_depth_maps), hr_depth_maps.shape[1], hr_depth_maps.shape[2]), dtype=float)

#for i in range(len(rgb_textures)):
textures[0] = rgb2gray(rgb_textures[0])

lr_depth_map_not_scaled = cv.resize(hr_depth_maps[0], dsize=(int(hr_depth_maps.shape[2]/4), int(hr_depth_maps.shape[1]/4)), interpolation=cv.INTER_CUBIC)
lr_depth_map = cv.resize(lr_depth_map_not_scaled, dsize=(hr_depth_maps.shape[2], hr_depth_maps.shape[1]), interpolation=cv.INTER_CUBIC)

hr_pcl = NyuPointCloud(hr_depth_maps[0])
hr_pcl.create_ply("NYU-hr-ptcloud")
lr_pcl = NyuPointCloud(lr_depth_map)
lr_pcl.create_ply("NYU-lr-ptcloud")
hr_pcl_raw = NyuPointCloud(raw_depth_maps[0])
hr_pcl_raw.create_ply("NYU-hr-ptcloud-raw")


cmap = mpl.cm.get_cmap("winter").copy()
cmap.set_under(color='black')

plt.figure('RGB')
plt.imshow(rgb_textures[0])
plt.figure('RAW DEPTH')
plt.imshow(raw_depth_maps[0], cmap=cmap)
plt.figure('GRAY')
plt.imshow(textures[0], cmap='gray')
plt.figure('HR DEPTH')
plt.imshow(hr_depth_maps[0], cmap=cmap)
plt.figure('LR DEPTH NOT SCALED')
plt.imshow(lr_depth_map_not_scaled, cmap=cmap)
plt.figure('LR DEPTH')
plt.imshow(lr_depth_map, cmap=cmap)

plt.show()
'''

def create_dataset(name, def_maps=True, canny=True):
  print("--- CREATE DATASET: " + name + " ---")
  data = dict()

  f = h5py.File('data/nyu_depth_v2_labeled.mat')
  depth_maps = f['depths']
  rgb_textures = f['images']

  rgb_textures = np.transpose(rgb_textures, (0, 3, 2, 1))
  hr_depth_maps = np.transpose(depth_maps, (0, 2, 1))

  textures = np.empty((len(rgb_textures), hr_depth_maps.shape[1], hr_depth_maps.shape[2]), dtype=float)
  lr_depth_maps = np.empty((len(hr_depth_maps), hr_depth_maps.shape[1], hr_depth_maps.shape[2]), dtype=float)

  print("===> Loading HR depth maps")
  data['hr'] = hr_depth_maps

  if def_maps:
    def_maps = np.empty((len(data["hr"]), data["hr"][0].shape[0], data["hr"][0].shape[1]), dtype=float)
    for i in range(len(data["hr"])):
      def_maps[i] = (data["hr"][i] > 0).astype(float)
    data['dm'] = def_maps

  print("===> Loading LR depth maps")
  for i in range(len(hr_depth_maps)):
    lr_depth_map_not_scaled = cv.resize(hr_depth_maps[i],
                                        dsize=(int(hr_depth_maps.shape[2] / 2), int(hr_depth_maps.shape[1] / 2)),
                                        interpolation=cv.INTER_CUBIC)
    lr_depth_maps[i] = cv.resize(lr_depth_map_not_scaled,
                                 dsize=(hr_depth_maps.shape[2], hr_depth_maps.shape[1]),
                                 interpolation=cv.INTER_CUBIC)
  data['lr'] = lr_depth_maps

  print("===> Loading HR textures")
  for i in range(len(hr_depth_maps)):
    textures[i] = rgb2gray(rgb_textures[i])
  data['tx'] = textures

  print("===> Creating canny masks")
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

dataset_name = "NYU2-scale_2-canny"

create_dataset(dataset_name, def_maps=True, canny=True)

create_dataset_norm_data(dataset_name)
