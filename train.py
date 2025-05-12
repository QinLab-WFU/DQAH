# 2022.06.28-Changed for building CMT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# Modified from Fackbook, Deit
# jianyuan.guo@huawei.com
#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from utils.tools import *
from model.qformer import QFormer
from torch.autograd import Variable
import torch
import torch.optim as optim
import time
import timm
from loguru import logger
from timm.utils import AverageMeter, random_seed
from apex import amp
torch.multiprocessing.set_sharing_strategy('file_system')
from relative_similarity import *
from centroids_generator import *
import torch.nn.functional as F
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
from timm.scheduler import create_scheduler , CosineLRScheduler
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, ApexScaler
from loguru import logger
from loss.acmvh_loss import Acmvh_out
from loss.losses import bit_var_loss
from loss.FAST_HPP import HouseHolder
import q_utils as utils
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

import warnings
warnings.filterwarnings("ignore")  # Del ImageNet Warnings
import os
from qFormer_args import get_config as get_configs
from qFormer_args import parse_option
from loss.hypbird import margin_contrastive
def get_config():
    config = {
        "alpha": 0.1,
        'info':"[QFormer]",
        "step_continuation": 20,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
       "datasets": "mirflickr",
        "Label_dim" : 10, 
        "epoch": 100,
        "test_map": 0,
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5}},
        "save_path": "save/HashNet",
        "device": torch.device("cuda:0"),
        'test_device':torch.device("cuda:0"),
        "bit_list": [16],
        "img_size": 224
    }
    config = config_dataset(config)
    return config

def train_val(config, bit,args=None,configs=None):
    device = torch.device(config['device'])

    torch.manual_seed(3407)
    np.random.seed(3407)
    torch.cuda.manual_seed(3407)

    cudnn.benchmark = True

    print(f"Creating model: {args.cfg}")

    model = build_model(configs, args, bit)

    model.to(device)
    # model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    input_size = [1, 3, 224, 224]
    input = torch.randn(input_size).to(device)
    from torchprofile import profile_macs
    if 'asmlp' not in args.cfg:
        macs = profile_macs(model.eval(), input)
        print('model flops:', macs, 'input_size:', input_size)
        model.train()

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    # linear_scaled_lr = args.lr * args.batch_size  / 512.0
    args.lr = linear_scaled_lr
    print('learning rate: ', args.lr)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        print('Pretrained Weiught Loaded')
        checkpoint['model'].pop('head.weight')
        checkpoint['model'].pop('head.bias')
        model.load_state_dict(checkpoint['model'], strict=False)
        print('\b\b\b\bLoaded Pretrained Model')

    model.to(config["device"])


    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    optimizer = config["optimizer"]["type"](model.parameters(), **(config["optimizer"]["optim_params"]))
    criterion = Acmvh_out().to(device)
    criterion2 = HouseHolder().to(device)
    criterion3 = bit_var_loss()
    Best_mAP = 0
    total_time = 0

    for epoch in range(config["epoch"]):
        # model.module.update_temperature()

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        logger.info("%s[%2d/%2d][%s] bit:%d, datasets:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["datasets"]), end="")



        model.train()
        train_loss = 0
        total_loss = 0
        start_time = time.time()
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            u,c = model(image)
            loss1 = criterion()
            loss2 = criterion3()
            train_loss = loss1 + loss2
            train_loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader)
        total_time += time.time() - start_time
        logger.info("\b\b\b\b\b\b\b Traintime: %.3f" % (total_time))



         

# /home/wbt/conda_env_with_new_amp/conda_env/anaconda3/bin/python3.9 CMT_train.py --output_dir './' --model cmt_s --batch-size 32 --apex-amp --input-size 224 --weight-decay 0.05 --drop-path 0.1 --epochs 300 --test_freq 100 --test_epoch 260 --warmup-lr 1e-7 --warmup-epochs 20
        logger.info("\b\b\b\b\b\b\b train_loss:%.4f" % ( train_loss))
        if (epoch + 1) > 0:
          Best_mAP, index_img = validate(config, Best_mAP, test_loader, dataset_loader, model, bit, epoch, 10)
          model.to(config["device"])


    
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(' traini ng and evaluation script', parents=[get_config()])
    # args = parser.parse_args()
    args,configs = parse_option()
    config = get_config()
    # 建立日志文件（Create log file）
    logger.add('logs/{time}' + config["info"] + '_' + config["datasets"] + ' alpha '+str(config["alpha"]) + '.log', rotation='50 MB', level='DEBUG')

    logger.info(config)
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/alexnet/HashNet_{config['datasets']}_{bit}.json"
        train_val(config, bit,args,configs)



