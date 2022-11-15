import logging
import argparse

import numpy as np
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import *
from unet import *
from torch import Tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

def pixel_error(pred: Tensor, gt: Tensor, t='all'):
  pred = pred.int()
  gt = gt.int()
  if t == 'all':
    cmp = pred != gt
  elif t == 'fake':
    cmp = pred > gt
  elif t == 'miss':
    cmp = pred < gt
  else:
    cmp = gt != gt
  cmp = cmp.int()
  return torch.count_nonzero(cmp).data.float()

def train(dataloader, model, optimizer, loss_function, history):
  model.to(torch.device(device))
  model.train()

  train_loss = 0
  bar = tqdm(total=len(dataloader.dataset), desc="Train")

  for batch, (lr_depth_map, texture, hr_depth_map) in enumerate(dataloader):
    lr_depth_map = lr_depth_map.to(torch.device(device))
    hr_depth_map = hr_depth_map.to(torch.device(device))

    optimizer.zero_grad()
    pred = model.forward(lr_depth_map.float())
    loss = loss_function(pred, hr_depth_map)
    loss.backward()
    optimizer.step()
    train_loss = loss
    bar.update(1)

  history.append(train_loss)
  bar.close()

  return train_loss


def test(dataloader, model, loss_function, history):
  model.to(torch.device(device))
  model.eval()

  test_loss = 0
  p_error = 0

  with torch.no_grad():
    for lr_depth_map, texture, hr_depth_map in tqdm(dataloader, desc="Test"):
      lr_depth_map = lr_depth_map.to(torch.device(device))
      hr_depth_map = hr_depth_map.to(torch.device(device))

      pred = model.forward(lr_depth_map.float())
      test_loss += loss_function(pred, hr_depth_map)
      p_error += pixel_error(pred, hr_depth_map)

  test_loss /= len(dataloader)
  p_error /= len(dataloader)
  history.append(test_loss)

  return test_loss


parser = argparse.ArgumentParser(description='Torch create HR defined pixels map')

parser.add_argument('--model', default='UNET', help='choose model')
parser.add_argument('--scale', type=int, default=4, help='scale factor')
parser.add_argument('--lr', default='0.00025', type=float, help='learning rate')
parser.add_argument('--result', default='./result_def_map', help='learning rate')
parser.add_argument('--epoch', default=100, type=int, help='max epoch')
opt = parser.parse_args()
print(opt)

s = datetime.now().strftime('%Y%m%d%H%M%S')
result_root = '%s/%s-scale_%s-model_%s-epochs_%s-lr_%s' % (opt.result, s,  opt.scale, opt.model, opt.epoch, opt.lr)

if not os.path.exists(result_root):
  os.makedirs(result_root)

logging.basicConfig(filename='%s/train.log' % result_root, format='%(asctime)s %(message)s', level=logging.INFO)

print('> Loading datasets')
dataset_name = 'lr-4-warior'

lr_transform = transforms.Compose([transforms.ToTensor()])
tx_transform = transforms.Compose([transforms.ToTensor()])
hr_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = DepthMapSRDataset(dataset_name, train=True, task='def_map')
test_dataset = DepthMapSRDataset(dataset_name, train=False, task='def_map')

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

loss = None
model = None
optimizer = None

if opt.model == 'UNET':
  loss = nn.BCEWithLogitsLoss()
  model = UNet(in_channels=1, out_channels=1).float()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.00025, weight_decay=0.0)

train_history = []
test_history = []

for i in range(opt.epoch):
  print("===> Epoch[{}/{}]".format(i+1, opt.epoch))
  logging.info("===> Epoch[{}/{}]".format(i+1, opt.epoch))
  train_loss = train(train_dataloader, model, optimizer, loss, train_history)
  test_loss = test(test_dataloader, model, loss, train_history)
  print(">>> Training Loss: {:.4f} ### Testing Loss: {:>8f}".format(train_loss, test_loss))
  logging.info(">>> Training Loss: {:.4f} ### Testing Loss: {:>8f}".format(train_loss, test_loss))

  if (i + 1) % 100 == 0:
    torch.save({
      'epoch': i,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': loss,
    }, result_root + "/checkpoint-epoch-" + str(i+1) + ".pt")

torch.save(model.state_dict(), result_root + "/trained_model.pt")

