import numpy as np
import os
import cv2 as cv
import torch
from torch.utils.data import Dataset
from unet import *
from tqdm import tqdm

class DepthMapSRDataset(Dataset):

  def __init__(self, name, lr_transform=None, tx_transform=None, hr_transform=None, dm_transform=None, train=True, train_part=0.7, task='depth_map_sr'):
    self.name = name
    self.lr_transform = lr_transform
    self.tx_transform = tx_transform
    self.hr_transform = hr_transform
    self.dm_transform = dm_transform
    self.train = train
    self.train_part = train_part
    self.task = task

    assert os.path.isfile('dataset/' + self.name + '.npy'), "Dataset '" + self.name + "' does not exist"
    self.data = np.load('dataset/' + self.name + '.npy', allow_pickle=True).tolist()

    shuffler = np.random.permutation(len(self.data["hr"]))
    self.data["hr"] = self.data["hr"][shuffler]
    self.data["lr"] = self.data["lr"][shuffler]
    self.data["tx"] = self.data["tx"][shuffler]

    self.train_data = {"lr": self.data["lr"][:int(len(self.data['tx']) * self.train_part)],
                       "tx": self.data["tx"][:int(len(self.data['tx']) * self.train_part)],
                       "hr": self.data["hr"][:int(len(self.data['tx']) * self.train_part)]}

    self.test_data = {"lr": self.data["lr"][int(len(self.data['tx']) * self.train_part):],
                      "tx": self.data["tx"][int(len(self.data['tx']) * self.train_part):],
                      "hr": self.data["hr"][int(len(self.data['tx']) * self.train_part):]}

    if self.task == 'def_map':
      for i in range(len(self.data["hr"])):
        self.data["hr"][i] = np.where(self.data["hr"][i] != 0, 1.0, 0)
    elif self.task == 'depth_map_sr':
      self.data["dm"] = self.data["dm"][shuffler]
      self.train_data["dm"] = self.data["dm"][:int(len(self.data['dm']) * self.train_part)]
      self.test_data["dm"] = self.data["dm"][int(len(self.data['dm']) * self.train_part):]

  def __len__(self):
    if self.train:
      return len(self.train_data['tx'])
    return len(self.test_data['tx'])


  def __getitem__(self, idx):
    if self.task == 'depth_map_sr':
      if self.train:
        sample = [self.train_data['lr'][idx], self.train_data['tx'][idx], self.train_data['hr'][idx], self.train_data['dm'][idx]]
      else:
        sample = [self.test_data['lr'][idx], self.test_data['tx'][idx], self.test_data['hr'][idx], self.test_data['dm'][idx]]
    else:
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

    if self.dm_transform and self.task == 'depth_map_sr':
      sample[3] = self.hr_transform(sample[3])

    if self.task == 'depth_map_sr':
      return sample[0], sample[1], sample[2], sample[3]
    else:
      return sample[0], sample[1], sample[2]

def create_dataset(name, hr_dir, lr_dir, textures_dir, scale_lr=False, def_maps=False):
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

  if def_maps:
    print("===> Computing def_maps")
    model = UNet(in_channels=1, out_channels=1).float()
    model.load_state_dict(torch.load("result_def_map/20221103183019-scale_4-model_UNET-epochs_100-lr_0.0005/trained_model.pt"))
    def_maps = np.empty((len(data["hr"]), data["hr"][0].shape[0], data["hr"][0].shape[1]), dtype=float)
    transform = torchvision.transforms.ToTensor()
    for i in tqdm(range(len(data["hr"])), desc="Computing def_maps"):
      input_tensor = transform(data["lr"][i])
      input_tensor = torch.unsqueeze(input_tensor, dim=0)
      with torch.no_grad():
        tensor_output = model.forward(input_tensor.float())
      act = nn.Sigmoid()
      tensor_output = act(tensor_output) > 0.5
      tensor_output = tensor_output.float()
      def_maps[i] = tensor_output[0][0].numpy()
    data['dm'] = def_maps

  print("===> SAVING .npy file ...")
  np.save("dataset/" + name + ".npy", data, allow_pickle=True)

  return data