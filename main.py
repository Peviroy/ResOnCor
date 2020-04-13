# make sure to change to the projict directory
import os
import sys
os.chdir(os.path.split(os.path.realpath(__file__))[0]) 
sys.path.append(os.path.abspath(".."))

# python embedded api
import time
import warnings 
import argparse

# pytorch api
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# own api
from models import ResNet18
from utils import accuracy_calc
from utils import draw_acc_loss
from data.dataset import CorelDataset
from utils.Meter import AverageMeter, ProgressMeter


parser = argparse.ArgumentParser(description="Resnet on CorelDataset")
parser.add_argument('--model-folder', default='./checkpoints', help='folder to save models', dest='model_folder')
parser.add_argument('--data', default='./dataset', help='where the data set is stored')
parser.add_argument('--batch', default=64, type=int, help='batch size of data input')
parser.add_argument('--epoch', default='5', type=int, help='the number of cycles to train the model')
parser.add_argument('--save', default='./', help='dir for saving document file')
parser.add_argument('--lr', default='0.01', type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay (default: 5e-4)', dest='weight_decay')
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
    if args.gpu is not None: 
        if not torch.cuda.is_available():     
            warnings.warn('The specific GPU is not found.')
            device = torch.device('cpu')
        else:
            print('Using gpu: {0} in device: {1}'.format(args.gpu, torch.cuda.get_device_name()))
            device = torch.device('cuda', args.gpu)
            
    main_worker(device, args)
    
    
def main_worker(device, args):
    global best_acc1 # global 
    
    # *Hpyer argument
    EPOCH = args.epoch
    BATCH_SIZE = args.batch
    LR = args.lr            # learning rate
    MOMENTUM = args.momentum
    WEIGHT_DECAY = args.weight_decay
    

    # *Data loading 
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet normalize
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # cifar10 normalize
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # ToTensor will make range [0, 255] -> [0.0,1.0], so Normalize should be placed behind ToTensor()
        normalize
    ])
        
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = CorelDataset('./dataset/train',
                                './dataset/train/train.txt',
                                transform=transform_train)
    train_dataloder = DataLoader(train_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=4)

    test_dataset = CorelDataset('./dataset/test',
                                './dataset/test/test.txt',
                                transform=transform_test)
    test_dataloder = DataLoader(test_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=4)
                                
    # *create model
    model = ResNet18().to(device)
    
    # TODO:resume frome a checkpoint 
    
    # *loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    # *Start traning or validate
    
    # TODO: validate mode
    
    # if not to validate it, train it
    with open(args.save + 'accuracy.txt', 'w') as acc_f:
        with open(args.save + 'log.txt', 'w') as log_f:
            # record info in every epoch
            loss_list = [] 
            train_accuracy_list = []
            test_accuracy_list = []
            for epoch in range(0, EPOCH):
                # the task of outputing grogress has been completed within the train and test function
                train_loss, train_acc1, train_logger = train(args, device, train_dataloder, model, criterion, optimizer, current_epoch=epoch)
                acc1, test_logger = test(args, device, test_dataloder, model, criterion, optimizer, current_epoch=epoch)
                # remember best acc@1 and save checkpoint
                if (acc1 > best_acc1) | ((epoch+1) % 10 == 0):
                    best_acc1 = max(acc1, best_acc1)
                    print('Saving model in epoch: {0:d}'.format(epoch+1))
                    torch.save(model.state_dict(), '{0:s}/model_{1:03d}_{2:.3f}'.format(args.model_folder, epoch+1, acc1))
            
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
    draw_acc_loss(EPOCH, train_acc=train_accuracy_list, train_loss=loss_list, test_acc=test_accuracy_list, savedir=args.save)        
            
    print('Saving Final model in epoch')
    torch.save(model.state_dict(), '{0:s}/model_final'.format(args.model_folder))
    print("Training Finish")
    

    
def train(args, device, train_dataloader, model_net, criterion, optimizer, current_epoch=0):
    # Some tool helping measure
    batch_time = AverageMeter(name='Time', fmt=':6.3f') # time spent in a batch
    losses = AverageMeter('Loss', ':.4e') 
    top1_acc = AverageMeter('Acc@1', ':6.2f') # top 1 accuracy
    top5_acc = AverageMeter('Acc@5', ':6.2f') # top 5 accuracy
    progress = ProgressMeter(   # log 
        len(train_dataloader),
        [batch_time, losses, top1_acc, top5_acc],
        prefix="[Train] Epoch: [{}]".format(current_epoch+1))
    
    # train mode
    model_net.train()
    
    end = time.time()
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
        losses.update(loss.detach().item(), n=image_batch.size(0)) # detach makes no grad, 
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
        progress.display(batch_cnt+1)
    return  losses.avg, top1_acc.avg.item(), progress.get(batch_cnt+1)
     
            
def test(args, device, test_dataloader, model_net, criterion, optimizer, current_epoch=0):
    # Some tool helping measure
    batch_time = AverageMeter(name='Time', fmt=':6.3f') # time spent in a batch
    top1_acc = AverageMeter('Acc@1', ':6.2f') # top 1 accuracy
    top5_acc = AverageMeter('Acc@5', ':6.2f') # top 5 accuracy
    progress = ProgressMeter(   # log 
        len(test_dataloader),
        [batch_time, top1_acc, top5_acc],
        prefix="[Test] Epoch: [{}]".format(current_epoch+1))

    model_net.eval()
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
            progress.display(batch_cnt+1)


    return top1_acc.avg.item(), progress.get(batch_cnt+1)


if __name__ == '__main__':
    main()