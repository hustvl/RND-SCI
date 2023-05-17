import os
import time
import datetime
import random
import scipy.io as scio

import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

import torch.utils.data as tud

from real_data import *
from utils import gen_log, time2file_name, checkpoint, real_par_args, count_param, get_elaspe_time, model_input_setting
from model import model_generator, loss_generator

try:
    from fvcore.nn.flop_count import FlopCountAnalysis
except:
    pass

opt = real_par_args()
opt = model_input_setting(opt)

# device
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# random seed
if opt.seed is None:
    opt.seed = np.random.randint(2**31)

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
if opt.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
result_path = opt.outf + date_time + '/result/'
model_path = opt.outf + date_time + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

# load training data
CAVE = prepare_data_cave(opt.data_path_CAVE, 3 if opt.debug else 205)
KAIST = prepare_data_KAIST(opt.data_path_KAIST, 3 if opt.debug else 30)

# load test data
HR_HSI = prepare_test_data(opt.test_path, 5)
mask_3d_shift, mask_3d_shift_s = generate_rnd_masks(opt.mask_path)

# model
model = model_generator(opt.method, opt.pretrained_model_path).cuda()
print("Model init.")

freq_loss, spa_loss = loss_generator(opt.method)
print("Loss init.")

# optimizing
if opt.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
elif opt.optim == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))

if opt.scheduler == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)

mse = torch.nn.MSELoss().cuda()
print("optimizer init.")


def train(epoch, loader_train, logger):
    epoch_loss = 0

    train_begin = torch.cuda.Event(enable_timing=True)
    train_end = torch.cuda.Event(enable_timing=True)
    forward_times = 0
    backward_times = 0

    start_time = time.time()
    model.train()
    for i, (y, input, label, Mask, Phi, Phi_s) in enumerate(loader_train):
        y, input, label, Phi, Phi_s = Variable(y), Variable(input), Variable(label), Variable(Phi), Variable(Phi_s)
        y, input, label, Phi, Phi_s = y.cuda(), input.cuda(), label.cuda(), Phi.cuda(), Phi_s.cuda()
        if 'rnd' in opt.method:
            model.compute_parm(Phi, Phi_s)
        input_mask = init_mask(Mask, Phi, Phi_s, opt.input_mask)
        optimizer.zero_grad()

        train_begin.record()
        if 'cst' in opt.method:
            model_out, diff_pred = model(y, input, input_mask)
            loss = torch.sqrt(spa_loss(model_out, label))
            diff_gt = torch.mean(torch.abs(model_out.detach() - label), dim=1, keepdim=True)  # [b,1,h,w]
            loss_sparsity = F.mse_loss(diff_gt, diff_pred)
            loss = loss + 2 * loss_sparsity
        else:
            model_out = model(y, input, input_mask)
            loss = torch.sqrt(spa_loss(model_out, label))

        if 'hdnet' in opt.method:
            fdl_loss = freq_loss(model_out, label)
            loss = loss + 0.7 * fdl_loss
        train_end.record()
        forward_times += get_elaspe_time(train_begin, train_end)

        epoch_loss += loss.data
        train_begin.record()
        loss.backward()
        train_end.record()
        backward_times += get_elaspe_time(train_begin, train_end)

        optimizer.step()

        if (i + 1) % (125) == 0:
            print('%4d %4d / %4d loss = %.10f time = %s' % (epoch, i, len(loader_train.dataset) // opt.batch_size, epoch_loss /
                                                            ((i + 1) * opt.batch_size), datetime.datetime.now()))

    elapsed_time = time.time() - start_time
    logger.info('epcoh = %4d , loss = %.10f , time = %4.2f s' %
                (epoch + 1, epoch_loss / len(loader_train.dataset), elapsed_time))
    logger.info("===> Epoch {} Complete: Avg. forward_time: {:.3f} backward_time: {:.3f} train_time: {:.3f}".format(
        epoch, forward_times, backward_times, (forward_times + backward_times)))


def test(epoch, HR_HSI, mask_3d_shift, mask_3d_shift_s, logger):
    pred = []
    model.eval()
    begin = time.time()

    for j in range(5):
        with torch.no_grad():
            meas = HR_HSI[:, :, j]
            y = Variable(torch.FloatTensor(meas.copy()))
            y = y.unsqueeze(0)
            y = torch.FloatTensor(y).cuda()

            meas = meas / meas.max()
            meas = torch.FloatTensor(meas)
            input = meas.unsqueeze(0)
            input = Variable(input)
            input = input.cuda()

            mask_3d_shift = mask_3d_shift.cuda()
            mask_3d_shift_s = mask_3d_shift_s.cuda()
            if 'rnd' in opt.method:
                model.compute_parm(mask_3d_shift, mask_3d_shift_s)
            out = model(y, input, (mask_3d_shift, mask_3d_shift_s))

            result = out
            result = result.clamp(min=0., max=1.)
            res = result.cpu().permute(2, 3, 1, 0).squeeze(3).numpy()
            pred.append(res)  # H W C

    end = time.time()
    logger.info('===> Epoch {}: time: {:.2f}'.format(epoch, (end - begin)))
    return pred


def FPS(HR_HSI, model, logger):
    test_begin = torch.cuda.Event(enable_timing=True)
    test_end = torch.cuda.Event(enable_timing=True)
    all_forward_times = 0
    all_load_data_times = 0
    all_times = 0
    sence = 0

    for i in range(1, 100):
        forward_times = 0
        load_data_times = 0
        for j in range(5):
            with torch.no_grad():
                test_begin.record()
                meas = HR_HSI[:, :, j]

                y = Variable(torch.FloatTensor(meas.copy()))
                y = y.unsqueeze(0)
                y = y.cuda()

                meas = meas / meas.max() * 0.8
                meas = torch.FloatTensor(meas)
                input = meas.unsqueeze(0)
                input = Variable(input)
                input = input.cuda()

                mask_3d_shift = mask_3d_shift.cuda()
                mask_3d_shift_s = mask_3d_shift_s.cuda()
                test_end.record()
                load_data_times += get_elaspe_time(test_begin, test_end)

                test_begin.record()
                _ = model(y, input, (mask_3d_shift, mask_3d_shift_s))
                test_end.record()
                forward_times += get_elaspe_time(test_begin, test_end)

        logger.info("===> Epoch {} Complete: forward_time: {:.3f} data_load_time: {:.3f}, all_time: {:.3f}".format(
            i, forward_times, load_data_times, forward_times + load_data_times))
        if i > 20:
            all_forward_times += forward_times
            all_load_data_times += load_data_times
            all_times += forward_times + load_data_times
            sence += 5

    logger.info("forward all time : {:.3f} , load all time :  {:.3f} , all time {:.3f}".format(
        all_forward_times, all_load_data_times, all_times))
    logger.info("FPS: {:.3f} , all_time_FPS {:.3f}".format(sence / all_forward_times, sence / all_times))


def main():
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))

    #Data and envs perpare:
    Dataset = dataset(opt, CAVE, KAIST)
    loader_train = tud.DataLoader(Dataset, num_workers=8, batch_size=opt.batch_size, shuffle=True)

    # caclulate flops and params
    try:
        params = count_param(model)

        (y, input, label, Mask, Phi, Phi_s) = Dataset[0]
        y, input, label, Phi, Phi_s = Variable(y.unsqueeze(0)), Variable(input.unsqueeze(0)), Variable(
            label.unsqueeze(0)), Variable(Phi.unsqueeze(0)), Variable(Phi_s.unsqueeze(0))

        y, input, label, Phi, Phi_s = y.cuda(), input.cuda(), label.cuda(), Phi.cuda(), Phi_s.cuda()
        input_mask = init_mask(Mask, Phi, Phi_s, opt.input_mask)
        if 'rnd' in opt.method:
            model.compute_parm(Phi, Phi_s)
        flops = FlopCountAnalysis(model, (y, input, input_mask))
        logger.info("parms:{}, GFLOPs: {}".format(params, flops.total() / 1e9 / opt.batch_size))
    except:
        pass

    # Train
    for epoch in range(1, opt.max_epoch + 1):

        train(epoch, loader_train, logger)
        pred = test(epoch, HR_HSI, mask_3d_shift, mask_3d_shift_s, logger)
        scheduler.step()

        name = result_path + '/' + 'Test_{}'.format(epoch) + '.mat'
        scio.savemat(name, {'pred': pred})
        checkpoint(model, epoch, model_path, logger)

    # test Frame rate
    FPS(HR_HSI, model, logger)


if __name__ == '__main__':
    main()
