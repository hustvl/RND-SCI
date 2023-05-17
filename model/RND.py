import torch
import torch.nn as nn
from ops import roll_optim


def A(x, Phi):
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y


def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


#---------- original implement  -------------
def shift_3d(inputs, step=2):
    [_, nC, _, _] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=step * i, dims=2)
    return inputs


def shift_back_3d(inputs, step=2):
    [_, nC, _, _] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=(-1) * step * i, dims=2)
    return inputs


#---------- implement with cuda -------------
def shift_3d_cuda(inputs, step=2, dim=3):
    inputs = roll_optim(inputs, step, dim)
    return inputs


def shift_back_3d_cuda(inputs, step=2, dim=3):
    inputs = roll_optim(inputs, -step, dim)
    return inputs


class RND(nn.Module):

    def __init__(self, model):
        super(RND, self).__init__()
        self.backbone = model

    def compute_parm(self, Phi, Phi_s):
        if Phi == None or Phi_s == None:
            raise NotImplementedError("Please input Phi, Phi_s.")
        self.Phi_mean = Phi / (Phi_s.unsqueeze(1) + 1e-7)  # [28,256,310]
        self.Phi = Phi  #[28,256,310]

    def compute_xr(self, y):
        # range value
        xr = At(y, self.Phi_mean)
        return xr

    def compute_xn(self, input):
        q = shift_3d_cuda(input)  # 3D shift
        xn = q - At(torch.sum(self.Phi * q, dim=1), self.Phi_mean)  # null space value
        return xn

    def forward(self, y, model_input, input_mask):
        xr = self.compute_xr(y)
        output = self.backbone(y, model_input, input_mask)

        # for cst
        _ = None
        if isinstance(output, tuple):
            output, _ = output

        if output.shape[-1] != xr.shape[-1] and output.shape[-1] == output.shape[-2]:  # xn -> [b,28,256,256]
            b, c, h, w = output.shape
            zero_pad = torch.zeros((b, c, h, xr.shape[-1] - w)).cuda()
            output = torch.cat([output, zero_pad], dim=-1)

        xn = self.compute_xn(output)

        x = shift_back_3d_cuda((xr + xn))[:, :, :, :xr.shape[-2]]

        if _ is None:
            return x
        else:
            return x, _
