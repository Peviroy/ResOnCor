import os
import random
import argparse
import time
import math
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from datasets import *
from utils import setup_seed, yolo_gt_creator, fcos_gt_creator

from utils.augmentations import SSDAugmentation
from utils.vocapi_evaluator import VOCAPIEvaluator

setup_seed(41724138)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('-v', '--version', default='yolo', help='yolo')
    parser.add_argument('-d', '--dataset', default='voc', help='voc or coco')
    parser.add_argument('-ms',
                        '--multi_scale',
                        action='store_true',
                        default=False,
                        help='use multi-scale trick')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('-no_wp',
                        '--no_warm_up',
                        action='store_true',
                        default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=1, help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--num_workers',
                        default=8,
                        type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int, default=10, help='interval between evaluations')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False, help='use tensorboard')
    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights/', type=str, help='Gamma update for SGD')

    return parser.parse_args()


def train():
    args = parse_args()

    cfg = train_cfg
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # device
    if args.cuda and torch.cuda.is_available():
        print('Use gpu')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # multi-scale
    if args.multi_scale and args.version == 'yolo':
        print('Use the multi-scale trick ...')
        train_size = 640
        val_size = cfg['min_dim']['yolo']
    elif args.version == 'yolo':
        train_size = cfg['min_dim']['yolo']
        val_size = cfg['min_dim']['yolo']
    else: # fcos
        train_size = cfg['min_dim']['fcos']
        val_size = cfg['min_dim']['fcos']

    # dataset and evaluator
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    print('Loading the dataset...')

    if args.dataset == 'voc':
        num_classes = 20
        dataset = VOCDataset(root=VOC_ROOT, transform=SSDAugmentation(train_size))

        evaluator = VOCAPIEvaluator(data_root=VOC_ROOT,
                                    img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size),
                                    labelmap=VOC_CLASSES)
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             collate_fn=detection_collate,
                                             num_workers=args.num_workers,
                                             pin_memory=True)

    # build model
    if args.version == 'yolo':
        from models.yolov1 import myYOLO
        model = myYOLO(device, input_size=train_size, num_classes=num_classes, trainable=True)
        print('Let us train yolo on the %s dataset ......' % (args.dataset))

    elif args.version == 'fcos':
        from models.tinyfcos import FCOS
        model = FCOS(device, input_size=train_size, num_classes=num_classes, trainable=True)
        print('Let us train yolo on the %s dataset ......' % (args.dataset))
    else:
        print('We only support yolo and fcos for now.')
        exit()

    model.to(device).train()

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        log_path = os.path.join('log/coco/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)

    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    max_epoch = cfg['max_epoch']
    epoch_size = len(dataset) // args.batch_size

    # start training loop
    t0 = time.time()
    global_iteration = 0

    for epoch in range(args.start_epoch, max_epoch):

        # use step lr
        if epoch in cfg['lr_epoch']:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)

        for iter_i, (images, targets) in enumerate(dataloader):
            global_iteration += 1
            # WarmUp strategy for learning rate
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow(
                        (iter_i + epoch * epoch_size) * 1. / (args.wp_epoch * epoch_size), 4)
                    # tmp_lr = 1e-6 + (base_lr-1e-6) * (iter_i+epoch*epoch_size) / (epoch_size * (args.wp_epoch))
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)

            # multi-scale trick
            if args.multi_scale:
                if iter_i % 10 == 0 and iter_i > 0:
                    # randomly choose a new size
                    train_size = random.randint(10, 19) * 32
                    model.set_grid(train_size)
                # interpolate
                images = torch.nn.functional.interpolate(images,
                                                         size=train_size,
                                                         mode='bilinear',
                                                         align_corners=False)

            # make train label
            targets = [label.tolist() for label in targets]
            if args.version == 'yolo':
                targets = yolo_gt_creator(input_size=train_size,
                                          stride=model.stride,
                                          label_lists=targets)
            else: # fcos
                targets = fcos_gt_creator(input_size=train_size,
                                          num_classes=args.num_classes,
                                          stride=model.stride,
                                          scale_thresholds=model.scale_thresholds,
                                          label_lists=targets)

            # to device
            images = images.to(device)
            targets = targets.to(device)

            # forward and loss
            if args.version == 'yolo':
                conf_loss, cls_loss, bbox_loss, total_loss = model(images, targets=targets)
            else: # fcos
                cls_loss, ctn_loss, bbox_loss, total_loss = model(images, targets=targets)

            # backprop
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # display
            if iter_i % 10 == 0:
                t1 = time.time()
                if args.tfboard:
                    # viz loss
                    if args.version == 'yolo':
                        writer.add_scalar('obj loss', conf_loss.item(), global_iteration)
                        writer.add_scalar('cls loss', cls_loss.item(), global_iteration)
                        writer.add_scalar('box loss', bbox_loss.item(), global_iteration)
                        writer.add_scalar('total loss', total_loss.item(), global_iteration)
                    else: # fcos
                        writer.add_scalar('cls loss', cls_loss.item(), global_iteration)
                        writer.add_scalar('ctn loss', ctn_loss.item(), global_iteration)
                        writer.add_scalar('box loss', bbox_loss.item(), global_iteration)
                        writer.add_scalar('total loss', total_loss.item(), global_iteration)

                if args.version == 'yolo':
                    print(
                        '[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                        '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                        %
                        (epoch + 1, max_epoch, iter_i, epoch_size, tmp_lr, conf_loss.item(),
                         cls_loss.item(), bbox_loss.item(), total_loss.item(), train_size, t1 - t0),
                        flush=True)
                else:
                    print(
                        '[Epoch: %d/%d][Iter: %d/%d][cls: %.4f][ctn: %.4f][box: %.4f][loss: %.4f][lr: %.6f][size: %d][time: %.6f]'
                        % (epoch, cfg['max_epoch'], iter_i, epoch_size, cls_loss.item(),
                           ctn_loss.item(), bbox_loss.item(), total_loss.item(), tmp_lr, train_size,
                           t1 - t0),
                        flush=True)

                t0 = time.time()

        # evaluation
        if (epoch + 1) % args.eval_epoch == 0:
            model.trainable = False
            model.set_grid(val_size)
            model.eval()

            # evaluate
            evaluator.evaluate(model)

            # convert to training mode.
            model.trainable = True
            model.set_grid(train_size)
            model.train()

        # save model
        if (epoch + 1) % 10 == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(),
                       os.path.join(path_to_save, args.version + '_' + repr(epoch + 1) + '.pth'))


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
