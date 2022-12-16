import numpy as np
#from dataset import DepthMapSRDataset
from torch.utils.data import DataLoader
from pointcloud import *
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import measure, segmentation
from alive_progress import alive_bar

device = "cuda" if torch.cuda.is_available() else "cpu"

def fill_depth_map(img, background_value):
  mask = np.where(img > 0, 0, 1).astype(np.uint8)
  uint_mask = np.where(img > 0, 0, 255).astype(np.uint8)
  rgb_mask = np.dstack((uint_mask, uint_mask, uint_mask))

  labels, holes = measure.label(mask, background=0, return_num=True)

  print("Filling " + str(holes) + " holes")
  with alive_bar(holes) as bar:
    for i in range(1, holes + 1):
      label = labels == i

      hole_border = segmentation.mark_boundaries(np.zeros(rgb_mask.shape),
                                                 label.astype(np.int32),
                                                 (1, 0, 0),
                                                 None,
                                                 'outer')[:, :, 0].astype(float)
      border_values = img * hole_border
      label_img_border_pixels = np.sum(label.astype(np.int32)[0, :]) + \
                                np.sum(label.astype(np.int32)[label.shape[0] - 1, :]) + \
                                np.sum(label.astype(np.int32)[1:(label.shape[1] - 1), 0]) + \
                                np.sum(label.astype(np.int32)[1:(label.shape[1] - 1), label.shape[1] - 1])

      background = label_img_border_pixels > 0

      if not background:
        mx = np.ma.masked_array(border_values, mask=border_values == 0)
        line_values = mx.max(1)
        hole = (labels == i).astype(float) * line_values[:, np.newaxis]
      else:
        hole = (labels == i).astype(float) * background_value

      bar()

      img = np.where(hole > 0, hole, img)

    return img

'''
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_name = 'lr-4-warior'
norm_file_path = 'dataset/' + dataset_name + '_norm.npy'

assert os.path.isfile(norm_file_path), "Normalization file for dataset '" + dataset_name + "' does not exist"
norm_data = np.load(norm_file_path, allow_pickle=True).tolist()
depth_max = norm_data["depth"][1]

dataset = DepthMapSRDataset(dataset_name, train=False, task='depth_map_sr', norm=False)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

lr_depth_map, texture, hr_depth_map = next(iter(dataloader))
unfilled_hr_depth_map = hr_depth_map.clone()

hr_depth_map.to(torch.device(device))
unfilled_hr_depth_map.to(torch.device(device))

img = hr_depth_map[0][0].float().numpy()
filled = fill_depth_map(img, depth_max + 1)

cmap = mpl.cm.get_cmap("winter").copy()
cmap.set_under(color='black')

hr_pcl_unfilled = PointCloud(unfilled_hr_depth_map[0][0].numpy())
hr_pcl_unfilled.create_ply("UNFILLED-hr-ptcloud-actual")

hr_pcl = PointCloud(filled)
hr_pcl.create_ply("FILLED-hr-ptcloud-actual")

plt.figure(plt.figure('HR Depth map UNFILLED'))
plt.imshow(unfilled_hr_depth_map[0][0], cmap=cmap, vmin=0.0001)

plt.figure(plt.figure('HR Depth map'))
plt.imshow(filled, cmap=cmap, vmin=0.0001)

plt.show()
'''