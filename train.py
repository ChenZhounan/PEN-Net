import sys
import argparse
import os
import time
import datetime

from torch import optim
import torch
from tensorboardX import SummaryWriter
from soft_dtw_cuda import SoftDTW
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import dataloader, DataLoader

from visualization import vistualize_weight_gradient
import model
from datasets import get_dataset, default_collate_fn_
from config import cfg, cfg_from_file, assert_and_infer_cfg

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', required=True,
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--pretrained_model', default='', dest='pretrained_model', required=False,
                        help='continue train model')
    parser.add_argument('--local_rank', default='distributed', dest='local_rank', required=False,
                        help='multi GPUs training')
    return parser.parse_args()
  
if __name__ == '__main__':
    """ load config """
    opt = parse_args()
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """ multi GPUs setting """
    if opt.local_rank == 'distributed':
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
    else:
        local_rank = int(opt.local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    """ linear schedule for multi GPUs"""
    INIT_LR = cfg.SOLVER.BASE_LR
    # cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR / cfg.NUM_GPUS * torch.cuda.device_count()
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR / cfg.NUM_GPUS
    warmup_delta = (cfg.SOLVER.BASE_LR - INIT_LR) / (cfg.SOLVER.WARMUP_ITERS + 0.000001)
    # cfg.SOLVER.MAX_ITER = cfg.SOLVER.MAX_ITER * cfg.NUM_GPUS // torch.cuda.device_count()
    cfg.SOLVER.MAX_ITER = cfg.SOLVER.MAX_ITER * cfg.NUM_GPUS
    # cfg.TRAIN.SNAPSHOT_ITERS = cfg.TRAIN.SNAPSHOT_ITERS * cfg.NUM_GPUS // torch.cuda.device_count()
    cfg.TRAIN.SNAPSHOT_ITERS = cfg.TRAIN.SNAPSHOT_ITERS * cfg.NUM_GPUS
    # cfg.TRAIN.SNAPSHOT_BEGIN = cfg.TRAIN.SNAPSHOT_BEGIN * cfg.NUM_GPUS // torch.cuda.device_count()
    cfg.TRAIN.SNAPSHOT_BEGIN = cfg.TRAIN.SNAPSHOT_BEGIN * cfg.NUM_GPUS
    # cfg.SOLVER.GRAD_L2_CLIP = cfg.SOLVER.GRAD_L2_CLIP / torch.cuda.device_count()
    cfg.SOLVER.GRAD_L2_CLIP = cfg.SOLVER.GRAD_L2_CLIP
    print('init lr={}, base lr={}, warmup_d={}'.format(INIT_LR, cfg.SOLVER.BASE_LR,
                                                       warmup_delta))
    """ prepare log file """
    if local_rank == 0:
        # 避免多进程重复写入，只在GPU0的进程进行日志和模型的写操作
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        log_dir = os.path.join(cfg.OUTPUT_DIR, os.path.basename(opt.cfg_file)[:-4], t)
        tboard_dir = os.path.join(log_dir, 'tboard')
        model_dir = os.path.join(log_dir, 'model')
        sample_dir = os.path.join(log_dir, 'sample')
        os.makedirs(tboard_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        tfb_summary = SummaryWriter(tboard_dir)
    """ set model, criterion and optimizer"""
    net = model.Net_tr_2d().to(device)
    if len(opt.pretrained_model) > 0:
        net.load_state_dict(torch.load(opt.pretrained_model))
        print('load pretrained model from {}'.format(opt.pretrained_model))
    optimizer = optim.Adam(net.parameters(), lr=cfg.SOLVER.BASE_LR)
    torch.optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=[1000000,1400000], gamma=0.5)
    if cfg.TRAIN.PTS_LOSS_TYPE == 'l1':
        ptrs_criterion = torch.nn.L1Loss(reduction='none')
    elif cfg.TRAIN.PTS_LOSS_TYPE == 'l2' or cfg.TRAIN.PTS_LOSS_TYPE == 'l2_sqrt':
        ptrs_criterion = torch.nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError
    CE_criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.Tensor([1., 5., 10.]).cuda())
    # CE_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    SDTW_criterion = SoftDTW(use_cuda=True, gamma=0.1)
    """ set dataloader"""
    train_dataset = get_dataset(cfg.DATA_LOADER.TYPE, is_train=True)
    print('\033[31mtype: %s\tlen: %d\033[0m' % (type(train_dataset), len(train_dataset)))
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.IMS_PER_BATCH,
                              shuffle=True,
                              sampler=None,
                              drop_last=True,
                              collate_fn=default_collate_fn_,
                              num_workers=cfg.DATA_LOADER.NUM_THREADS)
    """start training iterations"""
    train_loader_iter = iter(train_loader)
    for step in range(cfg.SOLVER.MAX_ITER):
        prev_time = time.time()
        try:
            data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            data = next(train_loader_iter)
        # prepare input
        imgs, labels = data['img'].cuda(), data['label'].cuda()
        bs = len(imgs)
        sos = torch.Tensor(bs * [[[0, 0, 1, 0, 0]]]).cuda()
        input_seq = torch.cat([sos, labels], dim=1)[:, :-1]  # NTC, pad SOS, remove EOS
        # forward
        
        # cal loss
        dx_gt, dy_gt, pen_data_gt = labels[:, :, 0], labels[:, :, 1], labels[:, :, 2:]
        valid_mask = 1 - pen_data_gt[:, :, -1]  # [N, T]  1,1,1,1,0,0,0,0
        preds = net(imgs.cuda(), input_seq.cuda()) #torch.Size([128, 1, 64, 64])  torch.Size([128, 148, 5])
        dx_pred, dy_pred, pen_data_pred = preds[:, :, 0], preds[:, :, 1], preds[:, :, 2:]
        
        dx_loss = ptrs_criterion(dx_pred, dx_gt) * valid_mask
        # debug = ptrs_criterion(dx_pred, dx_gt)# [32, 125] 和 valid_mask维度相同
        dy_loss = ptrs_criterion(dy_pred, dy_gt) * valid_mask
        xy_loss = dx_loss + dy_loss
        if cfg.TRAIN.PTS_LOSS_TYPE == 'l2_sqrt':
            xy_loss = torch.sqrt(xy_loss + 1e-7)
        xy_loss = xy_loss.sum() / valid_mask.sum()
        B, T, C = pen_data_pred.size()
        state_label = torch.argmax(pen_data_gt, dim=-1)
        state_loss = CE_criterion(pen_data_pred.view(B * T, C), state_label.view(-1)).mean()
        if cfg.SOLVER.SDTW:
            if cfg.SOLVER.SDTW_BS:  # 整个batch直接算，在末尾pad处会有误差
                SDTW_loss = (SDTW_criterion(labels, preds) / 6000.).mean()
                loss = 0.5 * xy_loss + state_loss + SDTW_loss
            else:
                SDTW_losses = []
                for label, pred in zip(labels, preds):
                    label_valid_len = min(torch.where(label[:, -1] == 1)[0]) + 1
                    label = label[:label_valid_len]
                    try:
                        # pred_valid_len = min(torch.where(pred[:, -1] == 1)[0]) + 1
                        pred_state = pred[:, -3:]
                        state_max_arg = torch.argmax(pred_state, dim=1)
                        pred_valid_len = min(torch.where(state_max_arg == 2)[0]) + 1
                    except:
                        pred_valid_len = len(pred)
                    pred = pred[:pred_valid_len]
                    if cfg.SOLVER.ABS_DTW:
                        seq1 = torch.cumsum(label[:, :2], dim=0).unsqueeze(0)
                        seq2 = torch.cumsum(pred[:, :2], dim=0).unsqueeze(0)
                    else:
                        seq1 = label[:, :2].unsqueeze(0)
                        seq2 = pred[:, :2].unsqueeze(0)
                    SDTW_loss = SDTW_criterion(seq1, seq2) / 6000.
                    SDTW_losses.append(SDTW_loss)
                SDTW_loss = sum(SDTW_losses) / len(SDTW_losses)
                loss = 0.5 * xy_loss + state_loss + SDTW_loss
                # loss = state_loss + SDTW_loss
        else:
            loss = 0.5 * xy_loss + state_loss
        # backward and update parameters
        net.zero_grad()
        loss.backward()
        if cfg.SOLVER.GRAD_L2_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
        optimizer.step()
        # log summary
        if local_rank == 0:
            if (step + 1) % 100 == 0:
                if cfg.SOLVER.SDTW:
                    dx_loss_, dy_loss_, state_loss_, SDTW_loss_, loss_, xy_loss_ = dx_loss.mean().item(), dy_loss.mean().item(), state_loss.item(), SDTW_loss.item(), loss.item(), xy_loss.item()
                    log_dict = {'dx': dx_loss_, 'dy': dy_loss_, 'state': state_loss_, 'sdtw_loss': SDTW_loss_,
                                'loss': loss_, 'location': xy_loss_}
                else:
                    dx_loss_, dy_loss_, state_loss_, loss_, xy_loss_ = dx_loss.mean().item(), dy_loss.mean().item(), state_loss.item(), loss.item(), xy_loss.item()
                    log_dict = {'dx': dx_loss_, 'dy': dy_loss_, 'state': state_loss_, 'loss': loss_, 'location': xy_loss_}
                tfb_summary.add_scalars("losses", log_dict, step)
                iter_left = cfg.SOLVER.MAX_ITER - step
                time_left = datetime.timedelta(seconds=iter_left * (time.time() - prev_time))
                terminal_log = 'iter:%d ' % step
                for k, v in log_dict.items():
                    terminal_log += '%s:%.3f ' % (k, v)
                terminal_log += 'ETA:%s\r\n' % str(time_left)
                sys.stdout.write(terminal_log)
                # print(terminal_log)
            if (step + 1) % cfg.SOLVER.VISUAL_ITEMS == 0:
                vistualize_weight_gradient(tfb_summary, net, step, cfg.SOLVER.VISUAL_WEIGHT, cfg.SOLVER.VISUAL_GRADIENT)
            if (step + 1) > cfg.TRAIN.SNAPSHOT_BEGIN and (step + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                print('debug, 跳过测试')
                # test_metrics = test_core(net, test_loader)
                # net.train()
                # tfb_summary.add_scalars("test", test_metrics, step)
                # model_path = '{}/iter{}_trainloss{}_lndtw{}_diou{}.pth'.format(model_dir, step,
                #                                                       loss.item(), test_metrics['lndtw'],
                #                                                       test_metrics['diou'])
                model_path = '{}/iter{}_trainloss{}.pth'.format(model_dir, step,
                                                                loss.item())
                torch.save(net.state_dict(), model_path)
                print('save model to {}'.format(model_path))
