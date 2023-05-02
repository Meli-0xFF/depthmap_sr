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
from tqdm import tqdm

import time


def main():
  dataset_name = "NEAREST-BIG-DATASET"

  dataset = DepthMapSRDataset(dataset_name, train=False, task='depth_map_sr', norm=True, gaussian_noise=False)
  dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

  model1 = FDSR_Net(num_feats=32, kernel_size=3, trainable_upsampling=False).float()
  model2 = DKN(kernel_size=3, filter_size=15, residual=True, trainable_upsampling=False).float()
  model1.load_state_dict(torch.load("result/20230404111617-model_FDSR-epochs_1000/trained_model.pt"))
  model2.load_state_dict(torch.load("result/20230412065236-model_DKN-epochs_100/trained_model.pt"))

  device = "cuda" if torch.cuda.is_available() else "cpu"

  model1.to(torch.device(device))
  model2.to(torch.device(device))

  object_loss1 = 0
  object_loss2 = 0
  object_loss3 = 0

  rmse_loss1 = 0
  rmse_loss2 = 0
  rmse_loss3 = 0

  obj_rmse_loss1 = 0
  obj_rmse_loss2 = 0
  obj_rmse_loss3 = 0

  mse_loss = torch.nn.MSELoss()
  n = Normalization(dataset_name)

  with torch.no_grad():
    for lr_depth_map, texture, hr_depth_map, def_map, object_mask in tqdm(dataloader, desc="Test"):

      lr_depth_map = lr_depth_map.to(torch.device(device))
      texture = texture.to(torch.device(device))
      hr_depth_map = hr_depth_map.to(torch.device(device))
      object_mask = object_mask.to(torch.device(device))
      def_map = def_map.to(torch.device(device))

      tensor_output1 = model1.forward((texture.float(), lr_depth_map.float()))
      tensor_output2 = model2.forward((torch.unsqueeze(torch.stack((texture[0][0], texture[0][0], texture[0][0])), dim=0).float(), lr_depth_map.float()))

      object_loss1 += metrics.object_loss(n.recover_normalized_sample(tensor_output1, "depth").float() * def_map, n.recover_normalized_sample(hr_depth_map, "depth").float() * def_map, object_mask.float()).item()
      object_loss2 += metrics.object_loss(n.recover_normalized_sample(tensor_output2, "depth").float() * def_map, n.recover_normalized_sample(hr_depth_map, "depth").float() * def_map, object_mask.float()).item()
      object_loss3 += metrics.object_loss(n.recover_normalized_sample(lr_depth_map, "depth").float() * def_map, n.recover_normalized_sample(hr_depth_map, "depth").float() * def_map, object_mask.float()).item()

      rmse_loss1 += torch.sqrt(mse_loss(n.recover_normalized_sample(tensor_output1, "depth") * def_map, n.recover_normalized_sample(hr_depth_map, "depth") * def_map)).item()
      rmse_loss2 += torch.sqrt(mse_loss(n.recover_normalized_sample(tensor_output2, "depth") * def_map,n.recover_normalized_sample(hr_depth_map, "depth") * def_map)).item()
      rmse_loss3 += torch.sqrt(mse_loss(n.recover_normalized_sample(lr_depth_map, "depth") * def_map, n.recover_normalized_sample(hr_depth_map, "depth") * def_map)).item()


      or1 = torch.sum(torch.square(n.recover_normalized_sample(tensor_output1, "depth") * object_mask - n.recover_normalized_sample(hr_depth_map, "depth") * object_mask) / torch.sum(object_mask))
      or2 = torch.sum(torch.square(n.recover_normalized_sample(tensor_output2, "depth") * object_mask - n.recover_normalized_sample(hr_depth_map, "depth") * object_mask) / torch.sum(object_mask))
      or3 = torch.sum(torch.square(n.recover_normalized_sample(lr_depth_map, "depth") * object_mask - n.recover_normalized_sample(hr_depth_map, "depth") * object_mask) / torch.sum(object_mask))

      obj_rmse_loss1 += torch.sqrt(or1).item()
      obj_rmse_loss2 += torch.sqrt(or2).item()
      obj_rmse_loss3 += torch.sqrt(or3).item()

    object_loss1 /= len(dataloader)
    object_loss2 /= len(dataloader)
    object_loss3 /= len(dataloader)

    rmse_loss1 /= len(dataloader)
    rmse_loss2 /= len(dataloader)
    rmse_loss3 /= len(dataloader)

    obj_rmse_loss1 /= len(dataloader)
    obj_rmse_loss2 /= len(dataloader)
    obj_rmse_loss3 /= len(dataloader)

  print("Object loss FDSR: " + str(object_loss1))
  print("Object loss DKN: " + str(object_loss2))
  print("Object loss NEAREST: " + str(object_loss3))

  print("RMSE loss FDSR: " + str(rmse_loss1))
  print("RMSE loss DKN: " + str(rmse_loss2))
  print("RMSE loss NEAREST: " + str(rmse_loss3))

  print("Object RMSE loss FDSR: " + str(obj_rmse_loss1))
  print("Object RMSE loss DKN: " + str(obj_rmse_loss2))
  print("Object RMSE loss NEAREST: " + str(obj_rmse_loss3))

if __name__ == "__main__":
  main()