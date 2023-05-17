from simu_data import init_mask, LoadTest, init_meas, generate_rnd_masks
import torch
from ops import roll_optim

#---------- data --------------
data_root = '/home/jywang/datasets/HSI/'
data_path = f"{data_root}/cave_1024_28/"
mask_path = f"{data_root}/TSA_simu_data/"
test_path = f"{data_root}/TSA_simu_data/Truth/"
input_mask = 'Phi_PhiPhiT'
input_setting = 'Y'

mask3d_batch_test, _ = init_mask(mask_path, input_mask, 10)  # -> 10 28 256 256
input_mask_test = generate_rnd_masks(mask_path)  # (1 28 256 310) , (1 256 310)

mask3d_batch_test = mask3d_batch_test.cuda().contiguous()
test_data = LoadTest(test_path)
test_gt = test_data.cuda().float().contiguous()
meas, input_meas = init_meas(test_gt, mask3d_batch_test, input_setting)  # meas == input_meas


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


def A(x, Phi):
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y


def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


out_shape = meas.shape[-2]
Phi, Phi_s = input_mask_test
Phi = Phi.cuda()
Phi_s = Phi_s.cuda()

Phi_mean = Phi / (Phi_s + 1e-7)

print(f'Phi:{Phi_mean.shape}, input:{input_meas.shape}')
xr = At(input_meas, Phi_mean)  # range value
xr = shift_back_3d(xr)

#------- check Axr=y ---------
xr_ = xr[:, :, :, 0:out_shape]
print(f'test_gt:{test_gt.shape}, xr:{xr_.shape}')

y, output_meas = init_meas(xr_.contiguous(), mask3d_batch_test, input_setting)
print(torch.sum(torch.abs(y - meas)))

# ----- check Axn = 0------
torch.manual_seed(42)
q = torch.rand(xr.shape).cuda()
xn = q - At(torch.sum(Phi * q, dim=1), Phi_mean)
xn_ = shift_back_3d_cuda(xn)
xn_ = xn_[:, :, :, 0:out_shape]

y, output_meas = init_meas(xn_.contiguous(), mask3d_batch_test, input_setting)
print(torch.sum(torch.abs(output_meas)))

# ----- check A(xr+xn) = y------
q = torch.rand(xr.shape).cuda()
xn = q - At(torch.sum(Phi * q, dim=1), Phi_mean)
x = xr + shift_back_3d_cuda(xn)
x_ = x[:, :, :, 0:out_shape]

y, output_meas = init_meas(x_.contiguous(), mask3d_batch_test, input_setting)
print(torch.sum(torch.abs(y - meas)))