import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

from dataset import *
from sr_models.fdsr import FDSR_Net
from sr_models.dkn import DKN
from sr_models.dct import DCTNet
from torch.utils.data import DataLoader
from evaluation.pointcloud import *
from data_preparation.normalization import Normalization
import metrics


def main():
  dataset_name = "warior-scale_4-filled-with_canny"

  dataset = DepthMapSRDataset(dataset_name, train=False, task='depth_map_sr', norm=True)
  dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

  lr_depth_map, texture, hr_depth_map, def_map, canny_mask = next(iter(dataloader))
  #texture = torch.unsqueeze(torch.stack((texture[0][0], texture[0][0], texture[0][0])), dim=0)
  model = FDSR_Net(num_feats=32, kernel_size=3).float()
  #model = DKN(kernel_size=3, filter_size=15, residual=True).float()
  #model = DCTNet()
  model.load_state_dict(torch.load("result/20230104185858-model_FDSR-epochs_1000/trained_model.pt"))

  with torch.no_grad():
    tensor_output = model.forward((texture.float(), lr_depth_map.float()))
    canny_loss = metrics.canny_loss(tensor_output, hr_depth_map, canny_mask).item()
    print(torch.max(tensor_output))
    tensor_output = torch.where(hr_depth_map >= 0, tensor_output, -1.0)

  l1_loss = torch.nn.L1Loss()

  print("L1 Loss: " + str(l1_loss(tensor_output, hr_depth_map).item()))
  print("Canny loss: " + str(canny_loss))

  n = Normalization('lr-4-warior-FILLED')

  lr_pcl = PointCloud(n.recover_normalized_sample(lr_depth_map, "depth")[0][0].numpy())
  hr_pcl = PointCloud(def_map[0][0].numpy() * n.recover_normalized_sample(hr_depth_map, "depth")[0][0].numpy())
  out_pcl = PointCloud(def_map[0][0].numpy() * n.recover_normalized_sample(tensor_output, "depth")[0][0].numpy())

  lr_pcl.create_ply("lr-ptcloud")
  hr_pcl.create_ply("hr-ptcloud")
  out_pcl.create_ply("out-ptcloud")

  cmap = mpl.cm.get_cmap("winter").copy()
  cmap.set_under(color='black')

  plt.figure('LR Depth map')
  plt.imshow(n.recover_normalized_sample(lr_depth_map, "depth")[0][0], cmap=cmap, vmin=500.0)

  plt.figure('HR Texture')
  plt.imshow(n.recover_normalized_sample(texture, "guide")[0][0], cmap='gray')

  plt.figure('HR Def map')
  plt.imshow(def_map[0][0], cmap='gray')

  plt.figure('HR Canny mask')
  plt.imshow(canny_mask[0][0], cmap='gray')

  plt.figure(plt.figure('HR Depth map'))
  plt.imshow(n.recover_normalized_sample(hr_depth_map , "depth")[0][0] * def_map[0][0], cmap=cmap, vmin=0.0000001)

  plt.figure(plt.figure('Predicted HR Depth map'))
  plt.imshow(n.recover_normalized_sample(tensor_output, "depth")[0][0] * def_map[0][0], cmap=cmap, vmin=500.0)

  plt.show()

if __name__ == "__main__":
  main()