import torch
import numpy as np
from torch import Tensor

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