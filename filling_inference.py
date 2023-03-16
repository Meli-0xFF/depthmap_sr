from dataset import DepthMapSRDataset
from torch.utils.data import DataLoader
from evaluation.pointcloud import *
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from object_filling import fill_depth_map, fill_texture
import cupy as cp

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_name = 'NEAREST-LED-WARIOR-scale_4-ACTUAL'
norm_file_path = 'dataset/' + dataset_name + '_norm.npy'

assert os.path.isfile(norm_file_path), "Normalization file for dataset '" + dataset_name + "' does not exist"
norm_data = np.load(norm_file_path, allow_pickle=True).tolist()
depth_max = norm_data["depth"][1]

dataset = DepthMapSRDataset(dataset_name, train=False, task='depth_map_sr', norm=False)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

lr_depth_map, texture, hr_depth_map, def_map, object_mask = next(iter(dataloader))
unfilled_hr_depth_map = (hr_depth_map * def_map).clone()

img = np.array((hr_depth_map * def_map)[0][0].float().numpy())
tex = np.array((texture / torch.max(texture))[0][0].float().numpy())
filled, object_map = fill_depth_map(img, depth_max)
filled_texture = fill_texture(tex, def_map[0][0].float().numpy())

cmap = mpl.cm.get_cmap("winter").copy()
cmap.set_under(color='black')

hr_pcl_unfilled = PointCloud(unfilled_hr_depth_map[0][0].numpy())
hr_pcl_unfilled.create_ply("UNFILLED-PLANE-hr-ptcloud-actual")

hr_pcl = PointCloud(cp.asnumpy(filled))
hr_pcl.create_ply("FILLED-PLANE-hr-ptcloud-actual")

plt.figure(plt.figure('HR Depth map UNFILLED'))
plt.imshow(unfilled_hr_depth_map[0][0], cmap=cmap, vmin=0.0001)

plt.figure(plt.figure('HR Depth map'))
plt.imshow(cp.asnumpy(filled), cmap=cmap, vmin=0.0001)

plt.figure(plt.figure('HR Filled Texture'))
plt.imshow(cp.asnumpy(filled_texture), cmap='gray')

plt.show()
