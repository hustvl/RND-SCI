import os
import datetime
import random

import torch
import numpy as np

from utils import simu_par_args, count_param, model_input_setting
from model import model_generator

# ArgumentParser
opt = simu_par_args()
opt = model_input_setting(opt)

# device
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# model
model = model_generator(opt.method, opt.pretrained_model_path if opt.resume_path is None else opt.resume_path, with_cp=opt.cp)
if 'rnd' in opt.method:
    Phi, Phis = torch.randn(1, 28, 256, 310).cuda(), torch.randn(1, 256, 310).cuda()
    model.compute_parm(Phi, Phis)
print("Model init.")


def Param_FLOPs_test():

    y = torch.rand(1, 256, 310).cuda()

    if 'H' == opt.input_setting or 'HM' == opt.input_setting:
        input_meas = torch.rand(1, 28, 256, 256).cuda()
    elif 'Y' == opt.input_setting:
        input_meas = torch.rand(1, 256, 310).cuda()
    else:
        raise ValueError(f"The options of input_setting are 'H', 'HM' and 'Y', but you input {opt.input_setting}.")

    #Phi, Phi_PhiPhiT, Mask or None
    if 'Mask' == opt.input_mask or None == opt.input_mask:
        input_mask = torch.rand(1, 28, 256, 256).cuda()
    elif 'Phi' == opt.input_mask:
        input_mask = torch.rand(1, 28, 256, 310).cuda()
    elif 'Phi_PhiPhiT' == opt.input_mask:
        input_mask = (torch.rand(1, 28, 256, 310).cuda(), torch.rand(1, 256, 310).cuda())
    else:
        raise ValueError(
            f"The options of input_mask are 'Phi', 'Phi_PhiPhiT', 'Mask' and None, but you input {opt.input_setting}.")

    from fvcore.nn.flop_count import FlopCountAnalysis
    flops = FlopCountAnalysis(model, (y, input_meas, input_mask))
    print("FLOPS total: {:.3f}".format(flops.total() / 1e9))

    print("Params: {}".format(count_param(model)))


def main():
    model.eval()
    Param_FLOPs_test()


if __name__ == '__main__':
    main()
