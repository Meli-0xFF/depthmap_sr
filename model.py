import torch
import torch.nn as nn
from tqdm import tqdm
from sr_models.fdsr import FDSR_Net
from sr_models.dkn import DKN
from sr_models.dct import DCTNet
from metrics import canny_loss

class Model:
  def __init__(self, name, train_dataloader, test_dataloader):
    self.name = name
    self.train_dataloader = train_dataloader
    self.test_dataloader = test_dataloader

    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.model = None
    self.optimizer = None
    self.loss_function = None
    self.scheduler = None

    if self.name == 'FDSR':
      self.loss_function = nn.L1Loss()
      self.model = FDSR_Net(num_feats=32, kernel_size=3).float()
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=0.0)

    elif self.name == 'DKN':
      self.loss_function = nn.L1Loss()
      self.model = DKN(kernel_size=3, filter_size=15, residual=True).cuda()
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
      self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.2)

    elif self.name == 'DCT':
      self.loss_function = nn.MSELoss()
      self.model = DCTNet()
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0)
      self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)

  def train(self):
    self.model.to(torch.device(self.device))
    self.model.train()

    train_loss = 0
    bar = tqdm(total=len(self.train_dataloader.dataset), desc="Train")

    for batch, (lr_depth_map, texture, hr_depth_map, def_map, canny_mask) in enumerate(self.train_dataloader):
      self.optimizer.zero_grad()

      if self.name == 'DKN' or self.name == 'DCT':
        texture = torch.unsqueeze(torch.stack((texture[0][0], texture[0][0], texture[0][0])), dim=0)

      lr_depth_map = lr_depth_map.to(torch.device(self.device))
      texture = texture.to(torch.device(self.device))
      hr_depth_map = hr_depth_map.to(torch.device(self.device))
      canny_mask = canny_mask.to(torch.device(self.device))

      pred = self.model.forward((texture.float(), lr_depth_map.float()))
      loss = self.loss_function(pred.float(), hr_depth_map.float()) + 2 * canny_loss(pred.float(), hr_depth_map.float(), canny_mask.float())
      loss.backward()

      self.optimizer.step()
      if self.scheduler is not None:
        self.scheduler.step()
        if self.optimizer.param_groups[0]['lr'] <= 1e-6:
          self.optimizer.param_groups[0]['lr'] = 1e-6

      train_loss += loss
      bar.update(1)

    train_loss /= len(self.test_dataloader)

    bar.close()

    return train_loss


  def test(self):
    self.model.to(torch.device(self.device))
    self.model.eval()

    test_loss = 0

    with torch.no_grad():
      for lr_depth_map, texture, hr_depth_map, def_map, canny_mask in tqdm(self.test_dataloader, desc="Test"):
        if self.name == 'DKN' or self.name == 'DCT':
          texture = torch.unsqueeze(torch.stack((texture[0][0], texture[0][0], texture[0][0])), dim=0)

        lr_depth_map = lr_depth_map.to(torch.device(self.device))
        texture = texture.to(torch.device(self.device))
        hr_depth_map = hr_depth_map.to(torch.device(self.device))
        canny_mask = canny_mask.to(torch.device(self.device))

        pred = self.model.forward((texture.float(), lr_depth_map.float()))
        test_loss += self.loss_function(pred, hr_depth_map) + 2 * canny_loss(pred.float(), hr_depth_map.float(), canny_mask.float())

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