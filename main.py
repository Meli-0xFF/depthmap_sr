import matplotlib.pyplot as plt
import matplotlib as mpl
from dataset import *
from fdsr import FDSR_Net
from torch.utils.data import DataLoader
from pointcloud import *
from normalization import Normalization
from metrics import pixel_error

def main():
  dataset_name = 'lr-4-warior-FILLED'

  '''
  create_dataset(dataset_name, hr_dir='data/led-warior/depth_map_out/',
                                lr_dir='data/led-warior/lr_4_depth_map_out/',
                                textures_dir='data/led-warior/texture_laser/',
                                scale_lr=True,
                                def_maps=True,
                                fill=True)

  '''

  dataset = DepthMapSRDataset(dataset_name, train=False, task='depth_map_sr', norm=True)
  dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

  lr_depth_map, texture, hr_depth_map, def_map = next(iter(dataloader))
  model = FDSR_Net(num_feats=32, kernel_size=3).float()
  model.load_state_dict(torch.load("result/20221213235622-model_FDSR-epochs_1000/trained_model.pt"))

  with torch.no_grad():
    tensor_output = model.forward((texture.float(), lr_depth_map.float()))
    tensor_output = torch.where(hr_depth_map >= 0, tensor_output, -1.0)

  l1_loss = torch.nn.L1Loss()
  l2_loss = torch.nn.MSELoss()
  print("L1 Loss: " + str(l1_loss(tensor_output, hr_depth_map).item()))
  print("L2 Loss: " + str(l2_loss(tensor_output, hr_depth_map).item()))
  print("Max Absolute Error: " + str(torch.max(torch.abs(hr_depth_map - tensor_output)).item()))
  print("Max Squared Error: " + str(torch.max(torch.pow(hr_depth_map - tensor_output, 2)).item()))

  n = Normalization('lr-4-warior-FILLED')

  lr_pcl = PointCloud(n.recover_normalized_sample(lr_depth_map, "depth")[0][0].numpy())
  hr_pcl = PointCloud(def_map[0][0].numpy() * n.recover_normalized_sample(hr_depth_map, "depth")[0][0].numpy())
  out_pcl = PointCloud(def_map[0][0].numpy() * n.recover_normalized_sample(tensor_output, "depth")[0][0].numpy())

  lr_pcl.create_ply("lr-ptcloud-actual")
  hr_pcl.create_ply("hr-ptcloud-actual")
  out_pcl.create_ply("out-ptcloud-actual")

  cmap = mpl.cm.get_cmap("winter").copy()
  cmap.set_under(color='black')

  plt.figure('LR Depth map')
  plt.imshow(n.recover_normalized_sample(lr_depth_map, "depth")[0][0], cmap=cmap, vmin=500.0)

  plt.figure('HR Texture')
  plt.imshow(n.recover_normalized_sample(texture, "guide")[0][0], cmap='gray')

  plt.figure('HR Def map')
  plt.imshow(def_map[0][0], cmap='gray')

  plt.figure(plt.figure('HR Depth map'))
  plt.imshow(n.recover_normalized_sample(hr_depth_map , "depth")[0][0] * def_map[0][0], cmap=cmap, vmin=0.0000001)

  plt.figure(plt.figure('Predicted HR Depth map'))
  plt.imshow(n.recover_normalized_sample(tensor_output, "depth")[0][0] * def_map[0][0], cmap=cmap, vmin=500.0)

  plt.show()

if __name__ == "__main__":
  main()