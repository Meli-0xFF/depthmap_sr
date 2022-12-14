import torch
import numpy as np
from torch import Tensor
import kornia
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import DataLoader
from dataset import *
from data_preparation.normalization import Normalization

def rmse(a: Tensor, b: Tensor):
  err = torch.mean((a.float() - b.float()) ** 2)
  return torch.sqrt(err).item()


def pixel_error(pred: Tensor, gt: Tensor, t='all'):
  pred = (pred >= 0).int()
  gt = (gt >= 0).int()
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


def canny_loss(pred: Tensor, gt: Tensor, canny_mask: Tensor):
  error = torch.sum(torch.abs(gt.float() - pred.float()) * canny_mask) / torch.count_nonzero(canny_mask).data.float()
  return error

def get_canny_mask(img: Tensor):
  img = img.to(torch.device("cuda"))
  canny = kornia.filters.Canny()
  gauss = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))
  _, edges = canny(img.float())
  canny_mask: torch.tensor = (gauss(edges.float()) > 0).float()
  return canny_mask