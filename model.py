import torch
import torch.nn as nn
from tqdm import tqdm
from fdsr import FDSR_Net


class Model:
  def __init__(self, name, train_dataloader, test_dataloader):
    self.name = name
    self.train_dataloader = train_dataloader
    self.test_dataloader = test_dataloader

    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.model = None
    self.optimizer = None
    self.loss_function = None

    if self.name == 'FDSR':
      self.loss_function = nn.L1Loss()
      self.model = FDSR_Net(num_feats=32, kernel_size=3).float()
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=0.0)

  def train(self):
    self.model.to(torch.device(self.device))
    self.model.train()

    train_loss = 0
    bar = tqdm(total=len(self.train_dataloader.dataset), desc="Train")

    for batch, (lr_depth_map, texture, hr_depth_map, def_map) in enumerate(self.train_dataloader):
      lr_depth_map = lr_depth_map.to(torch.device(self.device))
      texture = texture.to(torch.device(self.device))
      hr_depth_map = hr_depth_map.to(torch.device(self.device))
      def_map = def_map.to(torch.device(self.device))

      self.optimizer.zero_grad()
      pred = self.model.forward((texture.float(), lr_depth_map.float()))
      pred = pred * def_map
      loss = self.loss_function(pred, hr_depth_map)
      loss.backward()
      self.optimizer.step()
      train_loss = loss
      bar.update(1)

    bar.close()

    return train_loss


  def test(self):
    self.model.to(torch.device(self.device))
    self.model.eval()

    test_loss = 0

    with torch.no_grad():
      for lr_depth_map, texture, hr_depth_map, def_map in tqdm(self.test_dataloader, desc="Test"):
        lr_depth_map = lr_depth_map.to(torch.device(self.device))
        texture = texture.to(torch.device(self.device))
        hr_depth_map = hr_depth_map.to(torch.device(self.device))
        def_map = def_map.to(torch.device(self.device))

        pred = self.model.forward((texture.float(), lr_depth_map.float()))
        pred = pred * def_map
        test_loss += self.loss_function(pred, hr_depth_map)

    test_loss /= len(self.test_dataloader)

    return test_loss

  def save_checkpoint(self, dir, epoch):
    torch.save({
      'epoch': epoch,
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'loss': self.loss_function,
    }, dir + "/checkpoint-epoch-" + str(epoch + 1) + ".pt")

  def save_trained_model(self, dir):
    torch.save(self.model.state_dict(), dir + "/trained_model.pt")