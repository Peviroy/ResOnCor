# make sure to change to the projict directory
import os
import sys
os.chdir(os.path.split(os.path.realpath(__file__))[0]) 
sys.path.append(os.path.abspath(".."))

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch_backend import ResNet18
from torch_util import accuracy_calc
from torch_util import CorelDataset, AverageMeter, ProgressMeter

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from util import draw_acc_loss

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="Resnet on CorelDataset")
parser.add_argument('--model', default='./model', help='folder to save models')
parser.add_argument('--data', default='./dataset', help='where the data set is stored')
parser.add_argument('--batch', default=64, type=int, help='batch size of data input')
parser.add_argument('--epoch', default='5', help='the number of cycles to train the model')
parser.add_argument('--save', default='./', help='dir for saving document file')
parser.add_argument('--lr', default='0.01', type=float, help='learning rate')
args = parser.parse_args()

# *Hpyter argument
EPOCH = args.epoch
BATCH_SIZE = args.batch
LR = args.lr # learning rate

# *Pre processing
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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
                            

model = ResNet18().to(device)


            

def train(model_net, epochs, train_dataloder, test_dataloder):
    best_accracy = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    with open(savedir + 'accuracy.txt', 'w') as acc_f:
        with open(savedir + 'log.txt', 'w') as log_f:
            
            # record info in every epoch
            loss_list = [] 
            train_accuracy_list = []
            test_accuracy_list = []
            for epoch in range(epochs):
                print('In Train:')
                batch_time = AverageMeter(name='Time', fmt=':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                losses = AverageMeter('Loss', ':.4e')
                top1 = AverageMeter('Acc@1', ':6.2f')
                top5 = AverageMeter('Acc@5', ':6.2f')
                progress = ProgressMeter(
                    len(train_dataloder),
                    [batch_time, data_time, losses, top1, top5],
                    prefix="[Train] Epoch: [{}]".format(epoch+1))
                
                model_net.train()
                runing_loss = 0.0
                correct = 0.0
                total = 0.0
                end = time.time()
                for batch_cnt, batch_data in enumerate(train_dataloder):
                    data_time.update(time.time() - end)
                                        
                    length = len(train_dataloder)
                    image_batch, label_batch = batch_data['image'], batch_data['label']
                    image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                    
                    # compute output
                    outputs = model_net(image_batch)
                    loss = criterion(outputs, label_batch)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy_calc(outputs, label_batch, topk=(1, 5))
                    losses.update(loss.detach().item(), image_batch.size(0))
                    top1.update(acc1[0], image_batch.size(0))
                    top5.update(acc5[0], image_batch.size(0))
                    
                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    
                    # display
                    progress.display(batch_cnt+1)

                    # log
                    log_f.write(progress.get(batch_cnt+1))
                    log_f.write('\n')
                    log_f.flush()
                
                # One epoch train finish
                loss_list.append(losses.avg)
                train_accuracy_list.append(top1.avg)
                
                # Testing
                print('In Test:')
                batch_time = AverageMeter(name='Time', fmt=':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                top1 = AverageMeter('Acc@1', ':6.2f')
                top5 = AverageMeter('Acc@5', ':6.2f')
                progress = ProgressMeter(
                    len(test_dataloder),
                    [batch_time, data_time, top1, top5],
                    prefix="[Test]Epoch: [{}]".format(epoch+1))
                
                model_net.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    end = time.time()
                    for batch_cnt, data in enumerate(test_dataloder):
                        data_time.update(time.time() - end)
                        model_net.eval()
                        image_batch, label_batch = batch_data['image'], batch_data['label']
                        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                        # compute output
                        outputs = model_net(image_batch)

                        # measure accuracy and record loss
                        acc1, acc5 = accuracy_calc(outputs, label_batch, topk=(1, 5))
                        top1.update(acc1[0], image_batch.size(0))
                        top5.update(acc5[0], image_batch.size(0))

                        # measure elapsed time
                        batch_time.update(time.time() - end)
                        end = time.time()
                        
                        # display
                        progress.display(batch_cnt+1)

                        # log
                        acc_f.write(progress.get(batch_cnt+1))
                        acc_f.write('\n')
                        acc_f.flush()
                        
                    # One epoch test finish   
                    test_accuracy_list.append(top1.avg)
                    
                    # Model saving
                    if ((epoch+1) % 10 == 0) | (best_accracy < top1.avg.item()):
                        print('Saving model in epoch: {0:d}'.format(epoch+1))
                        torch.save(model_net.state_dict(), '{0:s}/model_{1:03d}_{2:.3f}'.format(args.model, epoch+1, top1.avg.item()))
                        best_accracy = top1.avg.item()
                
            # Epoch finished.
            draw_acc_loss(epochs, train_acc=train_accuracy_list, train_loss=loss_list, test_acc=test_accuracy_list, savedir=savedir)
                
                
            print('Saving Final model in epoch')
            torch.save(model_net.state_dict(), '{0:s}/model_final'.format(args.model))
            print("Training Finish")

savedir = args.save

def main():
    if not os.path.exists(args.data):
        print('No dataset scanned!')
        return
    if not os.path.exists(args.model):
        os.makedirs(args.model)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    print('Start Training\n')
    train(model, int(EPOCH), train_dataloder, test_dataloder)


if __name__ == '__main__':
    main()
    
    
