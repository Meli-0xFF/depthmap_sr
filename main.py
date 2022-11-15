import matplotlib.pyplot as plt
import matplotlib as mpl
from torchvision import transforms
from dataset import *
from fdsr import FDSR_Net
from torch.utils.data import DataLoader
from pointcloud import *

def main():
  dataset_name = 'lr-4-warior-with-maps'

  '''
  create_dataset(dataset_name, hr_dir='data/led-warior/depth_map_out/',
                                lr_dir='data/led-warior/lr_4_depth_map_out/',
                                textures_dir='data/led-warior/texture_laser/',
                                scale_lr=True,
                                def_maps=True)

  '''

  dataset = DepthMapSRDataset(dataset_name, train=False, task='depth_map_sr')
  dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

  lr_depth_map, texture, hr_depth_map, def_map = next(iter(dataloader))
  model = FDSR_Net(num_feats=32, kernel_size=3).float()
  model.load_state_dict(torch.load("result/20221106002135-scale_4-model_FSDR-epochs_100-lr_0.0005/trained_model.pt"))

  with torch.no_grad():
    tensor_output = model.forward((texture.float(), lr_depth_map.float())) * def_map

  lr_pcl = PointCloud(lr_depth_map[0][0].numpy())
  hr_pcl = PointCloud(hr_depth_map[0][0].numpy())
  out_pcl = PointCloud(tensor_output[0][0].numpy())

  lr_pcl.create_ply("lr-ptcloud-dm2")
  hr_pcl.create_ply("hr-ptcloud-dm2")
  out_pcl.create_ply("out-ptcloud-dm2")

  cmap = mpl.cm.get_cmap("winter").copy()
  cmap.set_under(color='black')

  plt.figure('LR Depth map')
  plt.imshow(lr_depth_map[0][0], cmap=cmap, vmin=0.0000001)

  plt.figure('HR Texture')
  plt.imshow(texture[0][0], cmap='gray')

  plt.figure(plt.figure('HR Depth map'))
  plt.imshow(hr_depth_map[0][0], cmap=cmap, vmin=0.0000001)

  plt.figure(plt.figure('Predicted HR Depth map'))
  plt.imshow(tensor_output[0][0], cmap=cmap, vmin=0.0000001)

  plt.figure(plt.figure('Predicted HR Defined pixel map'))
  plt.imshow(def_map[0][0], cmap='gray')

  plt.show()

if __name__ == "__main__":
  main()