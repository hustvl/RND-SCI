import re
import torch

from .TSANet import TSA_Net
from .MST import MST
from .CST import CST
from .HDNet import HDNet, FDL
from .DAUHST import DAUHST
from .SAUNetv1_5 import SAUNet

from .RND import RND


def saunet_param_setting(num_iterations, with_cp):
    bdpr, cdpr = 0, 0

    # Recommended hyper-parameters in simulation experiments in CAVE and KAIST datasets.
    if num_iterations == 1:
        bdpr, cdpr = 0, 0
    elif num_iterations == 2:
        bdpr, cdpr = 0.1, 0.1
    elif num_iterations == 3:
        bdpr, cdpr = 0.2, 0.1
    elif num_iterations == 5:
        bdpr, cdpr = 0.2, 0.1
    elif num_iterations == 9:
        bdpr, cdpr = 0.3, 0.0
    elif num_iterations == 13:
        bdpr, cdpr = 0.3, 0.1
    param = {
        'num_iterations': num_iterations,
        'cdpr': cdpr,
        'bdpr': bdpr,
        'num_blocks': [1, 1, 3],
        'cmb_kernel': 7,  # Convolutional Modulational Block kernel size, setting larger value with the pursuit of accuary
        'dw_kernel': 3,
        'ffn_ratio': 4,
        'with_cp': with_cp
    }
    return param


def model_generator(method, pretrained_model_path=None, with_cp=False):

    # original model
    if 'saunet' in method:
        num_iterations = int(re.findall("\d+", method)[0])
        param = saunet_param_setting(num_iterations, with_cp)
        model = SAUNet(**param).cuda()

    elif 'mst' in method:
        if 'mst_s' in method:
            model = MST(dim=28, stage=2, num_blocks=[2, 2, 2]).cuda()
        elif 'mst_m' in method:
            model = MST(dim=28, stage=2, num_blocks=[2, 4, 4]).cuda()
        elif 'mst_l' in method:
            model = MST(dim=28, stage=2, num_blocks=[4, 7, 5]).cuda()

    elif 'cst' in method:
        if 'cst_s' in method:
            model = CST(num_blocks=[1, 1, 2], sparse=True).cuda()
        elif 'cst_m' in method:
            model = CST(num_blocks=[2, 2, 2], sparse=True).cuda()
        elif 'cst_l' in method:
            model = CST(num_blocks=[2, 4, 6], sparse=True).cuda()
        elif 'cst_l_plus' in method:
            model = CST(num_blocks=[2, 4, 6], sparse=False).cuda()

    elif 'hdnet' in method:
        model = HDNet().cuda()

    elif 'dauhst' in method:
        num_iterations = int(re.findall("\d+", method)[0])
        model = DAUHST(num_iterations=num_iterations).cuda()

    elif 'tsa_net' in method:
        model = TSA_Net().cuda()
    else:
        raise (f'Method {method} is not defined !!!!')

    # RND for model
    if 'rnd' in method:
        model = RND(model)

    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path, map_location='cuda')
        if 'RND' in method:
            model.load_state_dict(
                {('' if 'backbone.' in k else 'backbone.') + k.replace('module.', ''): v
                 for k, v in checkpoint.items()},
                strict=True)
        else:
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()}, strict=True)

    return model


def loss_generator(method):
    freq_loss, loss = None, None

    if method == 'hdnet':
        fdl_loss = FDL(
            loss_weight=0.7,
            alpha=2.0,
            patch_factor=4,
            ave_spectrum=True,
            log_matrix=True,
            batch_matrix=True,
        ).cuda()
        freq_loss = fdl_loss
    else:
        loss = torch.nn.MSELoss().cuda()
    return freq_loss, loss
