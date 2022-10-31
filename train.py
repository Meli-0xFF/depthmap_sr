import torch
import torch.nn as nn
import logging
import argparse
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader
import models
from dataset import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(dataloader, model, optimizer, loss_function, history):
  model.to(torch.device(device))
  model.train()

  train_loss = 0
  bar = tqdm(total=len(dataloader.dataset), desc="Train")

  for batch, (lr_depth_map, texture, hr_depth_map) in enumerate(dataloader):
    lr_depth_map, texture, hr_depth_map = lr_depth_map.to(torch.device(device)), texture.to(torch.device(device)), hr_depth_map.to(torch.device(device))
    optimizer.zero_grad()
    pred = model.forward((texture.float(), lr_depth_map.float()))
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

  with torch.no_grad():
    for lr_depth_map, texture, hr_depth_map in tqdm(dataloader, desc="Test"):
      lr_depth_map, texture, hr_depth_map = lr_depth_map.to(torch.device(device)), texture.to(torch.device(device)), hr_depth_map.to(torch.device(device))
      pred = model.forward((texture.float(), lr_depth_map.float()))
      test_loss += loss_function(pred, hr_depth_map)

  test_loss /= len(dataloader)
  history.append(test_loss)

  return test_loss


parser = argparse.ArgumentParser(description='Torch FDSR')

parser.add_argument('--model', default='FSDR', help='choose model')
parser.add_argument('--scale', type=int, default=4, help='scale factor')
parser.add_argument('--lr', default='0.0005', type=float, help='learning rate')
parser.add_argument('--result', default='./result', help='learning rate')
parser.add_argument('--epoch', default=1, type=int, help='max epoch')

opt = parser.parse_args()
print(opt)

s = datetime.now().strftime('%Y%m%d%H%M%S')
result_root = '%s/%s-lr_%s-s_%s' % (opt.result, s, opt.lr, opt.scale)

if not os.path.exists(result_root):
  os.makedirs(result_root)

logging.basicConfig(filename='%s/train.log' % result_root, format='%(asctime)s %(message)s', level=logging.INFO)

print('> Loading datasets')
dataset_name = 'lr-4-warior'
norm = get_mean_and_std(dataset_name)
lr_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm[0], norm[1])])
tx_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm[2], norm[3])])
hr_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm[4], norm[5])])

train_dataset = DepthMapSRDataset(dataset_name, train=True,
                                      lr_transform=lr_transform,
                                      tx_transform=tx_transform,
                                      hr_transform=hr_transform)
test_dataset = DepthMapSRDataset(dataset_name, train=False,
                                      lr_transform=lr_transform,
                                      tx_transform=tx_transform,
                                      hr_transform=hr_transform)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

loss = None
model = None
optimizer = None

if opt.model == 'FSDR':
  loss = nn.L1Loss()
  model = models.FSDR_Net(num_feats=32, kernel_size=3).float()
  optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.0)

train_history = []
test_history = []

for i in range(opt.epoch):
  print("===> Epoch[{}/{}]".format(i+1, opt.epoch))
  train_loss = train(train_dataloader, model, optimizer, loss, train_history)
  test_loss = test(test_dataloader, model, loss, train_history)
  print(">>> Training Loss: {:.4f} ### Testing Loss: {:>8f}".format(train_loss, test_loss))

  if (i + 1) % 100 == 0:
    torch.save({
      'epoch': i,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': loss,
    }, result_root + "/checkpoint-epoch-" + str(i+1) + ".pt")

torch.save(model.state_dict(), result_root + "/trained_model.pt")