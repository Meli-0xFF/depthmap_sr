import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

from dataset import *
from sr_models.fdsr import FDSR_Net
from sr_models.dkn import DKN
from torch.utils.data import DataLoader
from evaluation.pointcloud import *
from data_preparation.normalization import Normalization
import metrics


def main():
  dataset_name = "NEAREST-INFERENCE"

  dataset = DepthMapSRDataset(dataset_name, train=False, task='depth_map_sr', norm=True, gaussian_noise=False)
  dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

  lr_depth_map, texture, hr_depth_map, def_map, object_mask = next(iter(dataloader))
  model1 = FDSR_Net(num_feats=32, kernel_size=3, trainable_upsampling=False).float()
  model2 = DKN(kernel_size=3, filter_size=15, residual=True, trainable_upsampling=False).float()
  model1.load_state_dict(torch.load("result/20230404111617-model_FDSR-epochs_1000/trained_model.pt"))
  model2.load_state_dict(torch.load("result/20230412065236-model_DKN-epochs_100/trained_model.pt"))

  device = "cuda" if torch.cuda.is_available() else "cpu"

  lr_depth_map = lr_depth_map.to(torch.device(device))
  texture = texture.to(torch.device(device))
  hr_depth_map = hr_depth_map.to(torch.device(device))
  object_mask = object_mask.to(torch.device(device))
  def_map = def_map.to(torch.device(device))

  model1.to(torch.device(device))
  model2.to(torch.device(device))

  with torch.no_grad():
    tensor_output1 = model1.forward((texture.float(), lr_depth_map.float()))
    tensor_output2 = model2.forward((torch.unsqueeze(torch.stack((texture[0][0], texture[0][0], texture[0][0])), dim=0).float(), lr_depth_map.float()))

    object_loss1 = metrics.object_loss(tensor_output1.float(), hr_depth_map.float(), object_mask.float())
    object_loss2 = metrics.object_loss(tensor_output2.float(), hr_depth_map.float(), object_mask.float())
    object_loss3 = metrics.object_loss(lr_depth_map.float(), hr_depth_map.float(), object_mask.float())

    tensor_output1 = torch.where(hr_depth_map >= 0, tensor_output1, -1.0)
    tensor_output2 = torch.where(hr_depth_map >= 0, tensor_output2, -1.0)

  mse_loss = torch.nn.MSELoss()

  n = Normalization(dataset_name)

  print("Object loss FDSR: " + str(object_loss1.item()))
  print("Object loss DKN: " + str(object_loss2.item()))
  print("Object loss NEAREST: " + str(object_loss3.item()))

  print("RMSE FDSR: " + str(torch.sqrt(mse_loss(tensor_output1 * def_map, hr_depth_map * def_map)).item()))
  print("RMSE DKN: " + str(torch.sqrt(mse_loss(tensor_output2 * def_map, hr_depth_map * def_map)).item()))
  print("RMSE NEAREST: " + str(torch.sqrt(mse_loss(lr_depth_map * def_map, hr_depth_map * def_map)).item()))

  print("RMSE FDSR real: " + str(torch.sqrt(mse_loss(n.recover_normalized_sample(tensor_output1, "depth") * def_map, n.recover_normalized_sample(hr_depth_map, "depth") * def_map)).item()))
  print("RMSE DKN real: " + str(torch.sqrt(mse_loss(n.recover_normalized_sample(tensor_output2, "depth") * def_map, n.recover_normalized_sample(hr_depth_map, "depth") * def_map)).item()))
  print("RMSE NEAREST real: " + str(torch.sqrt(mse_loss(n.recover_normalized_sample(lr_depth_map, "depth") * def_map, n.recover_normalized_sample(hr_depth_map, "depth") * def_map)).item()))

  lr_pcl = PointCloud((n.recover_normalized_sample(lr_depth_map, "depth") * def_map)[0][0].cpu().numpy())
  hr_pcl = PointCloud((n.recover_normalized_sample(hr_depth_map, "depth") * def_map)[0][0].cpu().numpy())
  out_pcl_fdsr = PointCloud((n.recover_normalized_sample(tensor_output1, "depth") * def_map)[0][0].cpu().numpy())
  out_pcl_dkn = PointCloud((n.recover_normalized_sample(tensor_output2, "depth") * def_map)[0][0].cpu().numpy())

  lr_pcl.create_ply("lr-ptcloud")
  hr_pcl.create_ply("hr-ptcloud")
  out_pcl_fdsr.create_ply("out-ptcloud-FDSR", clean=True)
  out_pcl_dkn.create_ply("out-ptcloud-DKN", clean=True)

  cmap = mpl.cm.get_cmap("winter").copy()
  cmap.set_under(color='black')

  plt.figure('Object map')
  plt.imshow(object_mask[0][0].cpu(),  cmap='gray')

  plt.figure('LR Depth map')
  plt.imshow(n.recover_normalized_sample(lr_depth_map, "depth")[0][0].cpu() * def_map[0][0].cpu(), cmap=cmap, vmin=0.0000001)

  plt.figure('HR Texture')
  plt.imshow(n.recover_normalized_sample(texture, "guide")[0][0].cpu(), cmap='gray')

  plt.figure(plt.figure('HR Depth map'))
  plt.imshow(n.recover_normalized_sample(hr_depth_map , "depth")[0][0].cpu() * def_map[0][0].cpu(), cmap=cmap, vmin=0.0000001)

  plt.figure(plt.figure('Predicted HR Depth map FDSR'))
  plt.imshow(n.recover_normalized_sample(tensor_output1, "depth")[0][0].cpu() * def_map[0][0].cpu(), cmap=cmap, vmin=0.0000001)

  plt.figure(plt.figure('Predicted HR Depth map DKN'))
  plt.imshow(n.recover_normalized_sample(tensor_output2, "depth")[0][0].cpu() * def_map[0][0].cpu(), cmap=cmap, vmin=0.0000001)

  plt.show()

if __name__ == "__main__":
  main()