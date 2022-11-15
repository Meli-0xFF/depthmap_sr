import logging
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from dataset import *
from model import Model


parser = argparse.ArgumentParser(description='Torch Depth map SR')

parser.add_argument('--model', default='FDSR', help='choose model')
parser.add_argument('--scale', type=int, default=4, help='scale factor') # fixed 4 for now
parser.add_argument('--lr', default='0.0005', type=float, help='learning rate')
parser.add_argument('--result', default='./result', help='learning rate')
parser.add_argument('--epoch', default=100, type=int, help='epochs')

opt = parser.parse_args()
print(opt)

s = datetime.now().strftime('%Y%m%d%H%M%S')
result_root = '%s/%s-scale_%s-model_%s-epochs_%s-lr_%s' % (opt.result, s,  opt.scale, opt.model, opt.epoch, opt.lr)

if not os.path.exists(result_root):
  os.makedirs(result_root)

logging.basicConfig(filename='%s/train.log' % result_root, format='%(asctime)s %(message)s', level=logging.INFO)

print('> Loading datasets')
dataset_name = 'lr-4-warior-with-maps'

train_dataset = DepthMapSRDataset(dataset_name, train=True, task='depth_map_sr')
test_dataset = DepthMapSRDataset(dataset_name, train=False, task='depth_map_sr')

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = Model(opt.model, train_dataloader, test_dataloader)

for i in range(opt.epoch):
  print("===> Epoch[{}/{}]".format(i+1, opt.epoch))
  logging.info("===> Epoch[{}/{}]".format(i+1, opt.epoch))
  train_loss = model.train()
  test_loss = model.test()
  print(">>> Training Loss: {:.4f} ### Testing Loss: {:>8f}".format(train_loss, test_loss))
  logging.info(">>> Training Loss: {:.4f} ### Testing Loss: {:>8f}".format(train_loss, test_loss))

  if (i + 1) % 100 == 0:
    model.save_checkpoint(result_root, i)

model.save_trained_model(result_root)