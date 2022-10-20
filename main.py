import matplotlib.pyplot as plt
import matplotlib as mpl
from torchvision import transforms
from dataset import *

def main():
  dataset_name = 'lr-4-warior'

  '''
  
  create_dataset(dataset_name, hr_dir='../data/led-warior/depth_map_out/',
                                lr_dir='../data/lr-4-led-warior/',
                                textures_dir='../data/led-warior/texture_laser/',
                                scale_lr=True)
  compute_mean_and_std(dataset_name)
  
  '''
  norm = get_mean_and_std(dataset_name)

  lr_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm[0], norm[1])])
  tx_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm[2], norm[3])])
  hr_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm[4], norm[5])])

  dsr_train_dataset = DepthMapSRDataset(dataset_name, train=True,
                                        lr_transform=lr_transform,
                                        tx_transform=tx_transform,
                                        hr_transform=hr_transform)

  #dsr_test_dataset = DepthMapSRDataset(dataset_name, train=False,
  #                                      lr_transform=lr_transform,
  #                                      tx_transform=tx_transform,
  #                                      hr_transform=hr_transform)

  cmap = mpl.cm.get_cmap("winter").copy()
  cmap.set_under(color='black')

  plt.figure('LR Depth map')
  plt.imshow(dsr_train_dataset.__getitem__(0)[0][0, :, :], cmap=cmap, vmin=0.0000001)

  plt.figure('HR Texture')
  plt.imshow(dsr_train_dataset.__getitem__(0)[1][0, :, :], cmap='gray')

  plt.figure(plt.figure('HR Depth map'))
  plt.imshow(dsr_train_dataset.__getitem__(0)[2][0, :, :], cmap=cmap, vmin=0.0000001)

  plt.show()

if __name__ == "__main__":
  main()