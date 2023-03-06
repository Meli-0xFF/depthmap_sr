import matplotlib.pyplot as plt
import torch
from torch import Tensor
import kornia
from dataset import *

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


def get_variance_image(img: Tensor):
  mean = torch.mean(img)
  var = torch.pow((img - mean), 2)
  var = var / torch.max(var)
  return var

def var_loss(pred: Tensor, gt: Tensor):
  var = get_variance_image(gt)
  return torch.sum(torch.abs(gt.float() - pred.float()) * var) / (gt.size(2) * gt.size(3))

def object_loss(pred: Tensor, gt: Tensor, object_mask: Tensor):
  weight = torch.where(object_mask < 0, 0.01, object_mask)
  return torch.sum(torch.pow((gt.float() - pred.float()) * weight, 2)) / (gt.size(2) * gt.size(3))