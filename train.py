import logging
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from dataset import *
from model import Model


parser = argparse.ArgumentParser(description='Torch Depth map SR')

parser.add_argument('--model', default='FDSR', help='choose model')
parser.add_argument('--result', default='./result', help='result dir')

opt = parser.parse_args()
print(opt)

epochs = 100
batch_size = 1

if opt.model == 'FDSR':
  epochs = 1000
  batch_size = 1
elif opt.model == 'DKN':
  epochs = 100
  batch_size = 1

s = datetime.now().strftime('%Y%m%d%H%M%S')
result_root = '%s/%s-model_%s-epochs_%s' % (opt.result, s, opt.model, epochs)

if not os.path.exists(result_root):
  os.makedirs(result_root)

logging.basicConfig(filename='%s/train.log' % result_root, format='%(asctime)s %(message)s', level=logging.INFO)

print('Loading datasets...')
dataset_name = 'lr-4-warior-with-maps'

train_dataset = DepthMapSRDataset(dataset_name, train=True, task='depth_map_sr')
test_dataset = DepthMapSRDataset(dataset_name, train=False, task='depth_map_sr')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = Model(opt.model, train_dataloader, test_dataloader)

for i in range(epochs):
  print("===> Epoch[{}/{}]".format(i+1, epochs))
  logging.info("===> Epoch[{}/{}]".format(i+1, epochs))
  train_loss = model.train()
  test_loss = model.test()
  print(">>> Training Loss: {:.4f} ### Testing Loss: {:>8f}".format(train_loss, test_loss))
  logging.info(">>> Training Loss: {:.4f} ### Testing Loss: {:>8f}".format(train_loss, test_loss))

  if (i + 1) % 100 == 0:
    model.save_checkpoint(result_root, i)

model.save_trained_model(result_root)