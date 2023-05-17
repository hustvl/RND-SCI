import torch
from ops import roll_optim


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


b = 2
c = 8
h = 3
w = 20
a1 = torch.randn((b, c, h, w), device="cuda")
a2 = a1.clone().detach()
a3 = a1.clone().detach()
a4 = a1.clone().detach()
out1 = shift_3d(a1, 2)
out2 = roll_optim(a2, 2, 3)
print(torch.sum(torch.abs(out1 - out2)))
out3 = shift_back_3d(a3, 2)
out4 = roll_optim(a4, -2, 3)
print(torch.sum(torch.abs(out3 - out4)))

Phi_batch = torch.randn((1, 28, 256, 310), device="cuda")
Phi_batch_ = Phi_batch.clone().detach()


def shift_3d_cuda(inputs, step=2, dim=3):
    inputs = roll_optim(inputs, 2, dim)
    return inputs


def shift_back_3d_cuda(inputs, step=-2, dim=3):
    inputs = roll_optim(inputs, -2, dim)
    return inputs


Phi_batch = shift_back_3d(Phi_batch)
Phi_batch[:, :, :, 256:] = 1e-7
Phi_batch = shift_3d(Phi_batch)

Phi_batch_ = shift_back_3d_cuda(Phi_batch_)
Phi_batch_[:, :, :, 256:] = 1e-7
Phi_batch_ = shift_3d_cuda(Phi_batch_)

print(torch.sum(torch.abs(Phi_batch - Phi_batch_)))
