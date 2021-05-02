# make sure to change to the projict directory
import warnings
from utils.torch_util import view_predicted
from utils import adjust_learning_rate
from utils.Meter import AverageMeter, ProgressMeter
from data.dataset import CorelDataset
from utils import draw_acc_loss
from utils import accuracy_calc
from models import resnet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch
import argparse
import time
import os
import sys
os.chdir(os.path.split(os.path.realpath(__file__))[0])
sys.path.append(os.path.abspath(".."))

parser = argparse.ArgumentParser(description="Resnet on CorelDataset")
parser.add_argument('--model-folder',
                    default='./checkpoints',
                    help='folder to save models',
                    dest='model_folder')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--official-pre',
                    default='',
                    type=str,
                    help='path to official pre-trained model',
                    dest='official')
parser.add_argument('--class-num',
                    default=10,
                    type=int,
                    help='number of classes classified',
                    dest='class_num')
parser.add_argument('--pre-epoch',
                    default=0,
                    type=int,
                    help='previous epoch (default: none)',
                    dest='pre_epoch')
parser.add_argument('--data', default='./dataset', help='where the data set is stored')
parser.add_argument('--batch', default=64, type=int, help='batch size of data input(default: 64)')
parser.add_argument('--epoch',
                    default=100,
                    type=int,
                    help='the number of cycles to train the model(default: 200)')
parser.add_argument('--save', default='./', help='dir for saving document file')
parser.add_argument('--lr', default='0.01', type=float, help='learning rate(default: 0.01)')
parser.add_argument('--lr-decay-step',
                    default='200',
                    type=int,
                    help='lr decayed by 10 every step',
                    dest='lr_decay_step')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum(default: 0.9)')
parser.add_argument('--weight-decay',
                    default=5e-4,
                    type=float,
                    help='weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
parser.add_argument('--validate', default=False, type=bool, help='validation mode')
parser.add_argument('--test', default=False, type=bool, help='test only mode')
parser.add_argument('--model', default='resnet18', type=str, help='resnet18、resnet34、resnet50，etc')
args = parser.parse_args()

best_acc1 = 0 # global


def main():
    """There we perform pre-processing work, for example, setting up GPU, prepare directory.
    """
    args = parser.parse_args()

    # Solve path problem
    if not os.path.exists(args.data):
        warnings.warn('No such dataset:')
        raise Exception("Invalid dataset path:", args.data)
    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # Only support single gpu training right now
    if not torch.cuda.is_available():
        warnings.warn('No GPU is not found. Use CPU')
        device = torch.device('cpu')
    else:
        print('Using gpu: {0} in device: {1}'.format(args.gpu, torch.cuda.get_device_name()))
        device = torch.device('cuda', args.gpu)
    main_worker(device, args)


def main_worker(device, args):
    global best_acc1 # global

    # *Hpyer argument
    EPOCH = args.epoch
    PRE_EPOCH = args.pre_epoch
    BATCH_SIZE = args.batch
    LR = args.lr # learning rate
    MOMENTUM = args.momentum
    WEIGHT_DECAY = args.weight_decay

    # *Data loading
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet normalize
    normalize_list = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
    normalize = transforms.Normalize(*normalize_list) # cifar10 normalize
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # ToTensor will make range [0, 255] -> [0.0,1.0], so Normalize should be placed behind ToTensor()
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = CorelDataset('./dataset/train',
                                 './dataset/train/train.txt',
                                 transform=transform_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    test_dataset = CorelDataset('./dataset/test',
                                './dataset/test/test.txt',
                                transform=transform_test)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # *create model
    if args.official: # official model's architecture is different
        # official model classify 1000 classes
        model = resnet(args.model, 1000)
        # load
        if os.path.isfile(args.resume):
            print('loading pretrained model: {0:s}'.format(args.resume))
            model.load_state_dict(torch.load(args.resume, map_location=device))
        # change fc
        fc_inputs = model.fc.in_features
        model.fc = nn.Linear(fc_inputs, args.class_num)
        model = model.to(device)
    else:
        model = resnet(args.model, args.class_num).to(device)

    # resume frome a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('loading pretrained model: {0:s}'.format(args.resume))
            model.load_state_dict(torch.load(args.resume, map_location=device))
        else:
            warnings.warn('No such checkpoint: {0:s}'.format(args.resume))

    # *loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # *Start traning or validate

    # validate mode: show image with its label and prediction
    if args.validate:
        # shuffle
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
        view_predicted(test_dataloader, model, device, normalization=normalize_list)
        return

    # test mode: get the accuracy of the model on test dataset
    elif args.test:
        print('Test mode')
        test(args, device, test_dataloader, model)
        return

    # if not to validate it, train it
    with open(args.save + 'accuracy.txt', 'w') as acc_f:
        with open(args.save + 'log.txt', 'w') as log_f:
            # record info in every epoch
            loss_list = []
            train_accuracy_list = []
            test_accuracy_list = []
            for epoch in range(PRE_EPOCH, EPOCH):
                print('In Train:')
                adjust_learning_rate(optimizer,
                                     epoch,
                                     original_lr=args.lr,
                                     decay_step=args.lr_decay_step)
                # the task of outputing grogress has been completed within the train and test function
                train_loss, train_acc1, train_logger = train(args,
                                                             device,
                                                             train_dataloader,
                                                             model,
                                                             criterion,
                                                             optimizer,
                                                             current_epoch=epoch)
                acc1, test_logger = test(args, device, test_dataloader, model, current_epoch=epoch)
                # remember best acc@1 and save checkpoint
                if (acc1 > best_acc1) | ((epoch + 1) % 10 == 0):
                    best_acc1 = max(acc1, best_acc1)
                    print('Saving model in epoch: {0:d}'.format(epoch + 1))
                    torch.save(
                        model.state_dict(),
                        '{0:s}/model_{1:03d}_{2:.3f}'.format(args.model_folder, epoch + 1, acc1))

                loss_list.append(train_loss)
                train_accuracy_list.append(train_acc1)
                test_accuracy_list.append(acc1)

                # # log
                log_f.write(train_logger)
                log_f.write('\n')
                log_f.flush()
                acc_f.write(test_logger)
                acc_f.write('\n')
                acc_f.flush()

    # Epoch finished.
    draw_acc_loss(PRE_EPOCH,
                  EPOCH,
                  train_acc=train_accuracy_list,
                  train_loss=loss_list,
                  test_acc=test_accuracy_list,
                  savedir=args.save)

    print('Saving Final model in epoch')
    torch.save(model.state_dict(), '{0:s}/model_final'.format(args.model_folder))
    print("Training Finish")


def train(args, device, train_dataloader, model_net, criterion, optimizer, current_epoch=0):

    batch_time = AverageMeter(name='Time', fmt=':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_acc = AverageMeter('Acc@1', ':6.2f') # top 1 accuracy
    top5_acc = AverageMeter('Acc@5', ':6.2f') # top 5 accuracy
    progress = ProgressMeter( # log
        len(train_dataloader), [batch_time, losses, top1_acc, top5_acc],
        prefix="[Train] Epoch: [{}]".format(current_epoch + 1))

    # train mode
    model_net.train()

    end = time.time()
    batch_cnt = 0
    for batch_cnt, batch_data in enumerate(train_dataloader):
        # move into specific device
        image_batch, label_batch = batch_data['image'], batch_data['label']
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)

        # output
        outputs = model_net(image_batch)

        # calculate loss
        loss = criterion(outputs, label_batch)

        # measure accuracy and record loss
        acc1, acc5 = accuracy_calc(outputs, label_batch, topk=(1, 5))
        # detach makes no grad,
        losses.update(loss.detach().item(), n=image_batch.size(0))
        top1_acc.update(acc1[0], n=image_batch.size(0))
        top5_acc.update(acc5[0], n=image_batch.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # display
        progress.display(batch_cnt + 1)
    return losses.avg, top1_acc.avg.item(), progress.get(batch_cnt + 1)


def test(args, device, test_dataloader, model_net, current_epoch=0):
    # Some tool helping measure
    print('In Test:')
    # time spent in a batch
    batch_time = AverageMeter(name='Time', fmt=':6.3f')
    top1_acc = AverageMeter('Acc@1', ':6.2f') # top 1 accuracy
    top5_acc = AverageMeter('Acc@5', ':6.2f') # top 5 accuracy
    progress = ProgressMeter( # log
        len(test_dataloader), [batch_time, top1_acc, top5_acc],
        prefix="[Test] Epoch: [{}]".format(current_epoch + 1))

    model_net.eval()
    batch_cnt = 0
    with torch.no_grad():
        end = time.time()
        for batch_cnt, batch_data in enumerate(test_dataloader):
            image_batch, label_batch = batch_data['image'], batch_data['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            outputs = model_net(image_batch)

            # measure accuracy and record loss
            acc1, acc5 = accuracy_calc(outputs, label_batch, topk=(1, 5))
            top1_acc.update(acc1[0], image_batch.size(0))
            top5_acc.update(acc5[0], image_batch.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # display
            progress.display(batch_cnt + 1)

    return top1_acc.avg.item(), progress.get(batch_cnt + 1)


if __name__ == '__main__':
    main()
