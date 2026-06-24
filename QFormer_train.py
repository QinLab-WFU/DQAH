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
from loss.loss import RelaHashLoss
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
from timm.scheduler import create_scheduler , CosineLRScheduler
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, ApexScaler
from loguru import logger
from loss.Hash_Loss import HashNetLoss
from loss.acmvh_loss import Acmvh_out
from loss.losses import bit_var_loss
from loss.FAST_HPP import HouseHolder
import QFormer_utils as utils
from loss.quadruplet_loss import QuadrupletMarginLoss
from loss.ada_loss import AdaQuadrupletLoss

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
# from cmt_args import get_args_parser
# from swiftformer_args import get_args_parser
from QFormer_args import get_config as get_configs
from QFormer_args import parse_option
from loss.hypbird import margin_contrastive
def build_model(config, args,bit):
    model_type = config.MODEL.TYPE
    if model_type == 'cross-scale':
        model = CrossFormer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.CROS.PATCH_SIZE,
                                in_chans=config.MODEL.CROS.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.CROS.EMBED_DIM,
                                depths=config.MODEL.CROS.DEPTHS,
                                num_heads=config.MODEL.CROS.NUM_HEADS,
                                group_size=config.MODEL.CROS.GROUP_SIZE,
                                mlp_ratio=config.MODEL.CROS.MLP_RATIO,
                                qkv_bias=config.MODEL.CROS.QKV_BIAS,
                                qk_scale=config.MODEL.CROS.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.CROS.APE,
                                patch_norm=config.MODEL.CROS.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                merge_size=config.MODEL.CROS.MERGE_SIZE,
                                hash_bit=bit)
    elif  model_type == 'qformer':
        model = QFormer(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN.EMBED_DIM,
                        depths=config.MODEL.SWIN.DEPTHS,
                        num_heads=config.MODEL.SWIN.NUM_HEADS,
                        window_size=config.MODEL.SWIN.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                        qk_scale=config.MODEL.SWIN.QK_SCALE,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN.APE,
                        patch_norm=config.MODEL.SWIN.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        coords_lambda=config.TRAIN.coords_lambda,
                        rpe=config.MODEL.QuadrangleAttention.rpe,
                        hash_bit=bit)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
def get_config():
    config = {
        "alpha": 0.1,

        'info':"[QFormer]",

        "step_continuation": 20,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
       "datasets": "mirflickr",
        #"datasets": "cifar10",
       # "datasets":'nuswide_21',
        #"datasets":'coco',
        "Label_dim" : 10, # number_class nuswide:21 coco:80 mirflickr:38
        "epoch": 150,
        "test_map": 0,
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5}},
        "save_path": "save/HashNet",
        "device": torch.device("cuda:1"),
        'test_device':torch.device("cuda:1"),
        "bit_list": [16,32,64,128],
        "img_size": 224,
        "patch_size": 4,
        "in_chans": 3, 
        "num_work": 10,
        "model_type": "QFormer",
        "top_img": 100,
        
        # RelaHash
        "Beta":8,
        'm':0.7,

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
    # optimizer = create_optimizer(args, model)

    # model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    # loss_scaler = ApexScaler()
    # print('Using NVIDIA APEX AMP. Training in mixed precision.')

    # lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        print('Pretrained Weiught Loaded')
        checkpoint['model'].pop('head.weight')
        checkpoint['model'].pop('head.bias')
        model.load_state_dict(checkpoint['model'], strict=False)
        print('\b\b\b\bLoaded Pretrained Model')

    model.to(config["device"])
    # model.cuda()
    # relative_similarity = RelativeSimilarity(nbit=bit, nclass=38, batchsize=config["batch_size"])
    # rela_optimizer = optim.Adam(relative_similarity.parameters(), lr = 1e-5)

    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    optimizer = config["optimizer"]["type"](model.parameters(), **(config["optimizer"]["optim_params"]))
    #criterion = HashNetLoss(config, bit)
    criterion = Acmvh_out(param_clf=1, param_sim=1).to(device)
    criterion2 = HouseHolder(bit).to(device)
    criterion3 = bit_var_loss()
    #criterion = QuadrupletMarginLoss(args, need_cnt=True).to(device)
   # criterion = AdaQuadrupletLoss(args, need_cnt=True).to(device)

    # quan_loss = RelaHashLoss(multiclass=True,beta=config['Beta'],m=config['m'])  # cifar10 _ False
    Best_mAP = 0

    for epoch in range(config["epoch"]):
        # model.module.update_temperature()

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        logger.info("%s[%2d/%2d][%s] bit:%d, datasets:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["datasets"]), end="")



        model.train()
        train_loss = 0
        total_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            u,c = model(image)
            loss1 = criterion(u,c, label.float())
            image_rot= F.normalize(criterion2(u.T).T)
            loss2 = criterion3 (image_rot)
            train_loss = loss1 + loss2
            train_loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader)



         

# /home/wbt/conda_env_with_new_amp/conda_env/anaconda3/bin/python3.9 CMT_train.py --output_dir './' --model cmt_s --batch-size 32 --apex-amp --input-size 224 --weight-decay 0.05 --drop-path 0.1 --epochs 300 --test_freq 100 --test_epoch 260 --warmup-lr 1e-7 --warmup-epochs 20
        logger.info("\b\b\b\b\b\b\b train_loss:%.4f" % ( train_loss))
        if (epoch + 1) > 2:
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



    # elif config['info'] == '[SwiftFormer]':
    #     print(f"Creating model: {config['model']}")
    #     net = create_model(
    #     config['model'],
    #     num_classes=config['Label_dim'],
    #     distillation=False,
    #     pretrained=config['mll'],
    #     fuse=config['eval'],
    #     hash_bit=config['bit']
    #     )
