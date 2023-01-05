import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import DepthMapSRDataset


def create_dataset_norm_data(dataset_name):
  norm_file_path = 'dataset/' + dataset_name + '_norm.npy'

  dataset = DepthMapSRDataset(dataset_name, train=False, task='depth_map_sr')
  dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

  bar = tqdm(total=len(dataloader.dataset), desc="Getting min and max")

  guide_interval = [1000000.0, 0.0]
  depth_interval = [1000000.0, 0.0]

  for batch, (lr_depth_map, texture, hr_depth_map, def_map, canny_mask) in enumerate(dataloader):
    guide_min = torch.min(texture).item()
    guide_max = torch.max(texture).item()

    depth_min = torch.min(hr_depth_map[hr_depth_map > 0.0]).item()
    depth_max = torch.max(hr_depth_map).item()

    if guide_min < guide_interval[0]:
      guide_interval[0] = guide_min

    if guide_max > guide_interval[1]:
      guide_interval[1] = guide_max

    if depth_min < depth_interval[0]:
      depth_interval[0] = depth_min

    if depth_max > depth_interval[1]:
      depth_interval[1] = depth_max

    bar.update(1)

  norm_data = {"guide": guide_interval,
               "depth": depth_interval}

  print(norm_data)

  np.save(norm_file_path, norm_data, allow_pickle=True)