from torch.utils.data import DataLoader
from dataset import *
from unet import *
import matplotlib as mpl
from torch import Tensor
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def visualize(pred: Tensor, gt: Tensor):
  gt = gt.int()
  pred = pred.int()
  fake = pred > gt
  fake = fake.int() * 255
  miss = gt > pred
  miss = miss.int() * 255

  return torch.stack((miss, fake, gt.int() * 255))


dataset_name = 'lr-4-warior'
print('> Loading datasets')

lr_transform = transforms.Compose([transforms.ToTensor()])
tx_transform = transforms.Compose([transforms.ToTensor()])
hr_transform = transforms.Compose([transforms.ToTensor()])

dataset = DepthMapSRDataset(dataset_name, train=False, task='def_map')

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = UNet(in_channels=1, out_channels=1).float()
lr_depth_map, texture, hr_depth_map = next(iter(dataloader))
model.load_state_dict(torch.load("result_defined_pixels_prediction/20221108231719-scale_4-model_UNET-epochs_100-lr_0.00025/trained_model.pt"))
print(lr_depth_map)
with torch.no_grad():
  tensor_output = model.forward(lr_depth_map.float())

act = nn.Sigmoid()
tensor_output = act(tensor_output) > 0.5
tensor_output = tensor_output.int()
print(tensor_output)

cmap = mpl.cm.get_cmap("winter").copy()
cmap.set_under(color='black')

plt.figure('LR Depth map')
plt.imshow(lr_depth_map[0][0], cmap=cmap, vmin=0.0000001)

plt.figure(plt.figure('HR Defined pixels map'))
plt.imshow(hr_depth_map[0][0], cmap='gray')

plt.figure(plt.figure('Predicted HR Defined pixel map'))
plt.imshow(tensor_output[0][0] > 0, cmap='gray')

plt.figure(plt.figure('Compare unet'))
plt.imshow(visualize(tensor_output[0][0], hr_depth_map[0][0]).permute(1, 2, 0), cmap='gray')

plt.figure(plt.figure('Compare bilinear'))
plt.imshow(visualize(torch.tensor(lr_depth_map[0][0] > 0, dtype=torch.int), hr_depth_map[0][0]).permute(1, 2, 0), cmap='gray')

plt.show()