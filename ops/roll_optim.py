import torch
from ops import _C


class _RollOptim(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, shift, dim):
    ctx.shift = shift
    ctx.dim = dim
    out = _C.roll_optim(input, shift, dim)
    return out

  @staticmethod
  @torch.autograd.function.once_differentiable
  def backward(ctx, dw):
    grad_in = _C.roll_optim(dw, -ctx.shift, ctx.dim)
    return grad_in, None, None


roll_optim = _RollOptim.apply
