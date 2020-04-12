# make sure to change to the projict directory
import os
import sys
os.chdir(os.path.split(os.path.realpath(__file__))[0]) 
sys.path.append(os.path.abspath(".."))

# ----------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch_backend import ResNet18
from torch_util import CorelDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from util import draw_acc_loss

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="Resnet on CorelDataset")
parser.add_argument('--model', default='./model', help='folder to save models')
parser.add_argument('--data', default='./dataset', help='where the data set is stored')
parser.add_argument('--resnet_type', default='resnet18', help='kind of resnet used to train. Eg.resnet18, resnet50')
parser.add_argument('--batch', default=32, help='batch size of data input')
parser.add_argument('--epoch', default='5', help='the number of cycles to train the model')
args = parser.parse_args()

# *Hpyter argument
EPOCH = args.epoch
BATCH_SIZE = args.batch
LR = 0.01  # learning rate

# *Pre processing
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # ToTensor will make range [0, 255] -> [0.0,1.0], so Normalize should be placed behind ToTensor()
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
    

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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
    with open('accuracy.txt', 'w') as acc_f:
        with open('log.txt', 'w') as log_f:
            loss_list = [] # record loss in every epoch
            train_accuracy_list = []
            test_accuracy_list = []
            for epoch in range(epochs):
                print('Epoch: {0:d} '.format(epoch+1))
                model_net.train()
                runing_loss = 0.0
                correct = 0.0
                total = 0.0

                print('Training')
                for batch_cnt, batch_data in enumerate(train_dataloder):
                    length = len(train_dataloder)
                    image_batch, label_batch = batch_data['image'], batch_data['label']
                    image_batch, label_batch = image_batch.to(device), label_batch.to(device)

                    optimizer.zero_grad()

                    outputs = model_net(image_batch)
                    loss = criterion(outputs, label_batch)
                    loss.backward()
                    optimizer.step()

                    runing_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += label_batch.size(0)
                    correct += (predicted == label_batch).sum().item()

                    result = '[Train][Epoch:{0:d}, iter:{1:d}] Loss: {2:.03f} | Acc: {3:.3f}%%'.format(
                        epoch+1, (batch_cnt+1+epoch*length), runing_loss / (batch_cnt+1), correct / total * 100.0
                    )
                    print(result)
                    log_f.write(result)
                    log_f.write('\n')
                    log_f.flush()
                # Trainning finished, mark down the loss and accuray of the whole epoch
                loss_list.append(runing_loss/len(train_dataloder))
                train_accuracy_list.append(100.0 * correct / len(train_dataloder.dataset))
                
                
                print('Testing')
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in test_dataloder:
                        model_net.eval()
                        image_batch, label_batch = batch_data['image'], batch_data['label']
                        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                        outputs = model_net(image_batch)

                        _, predicted = torch.max(outputs.data, 1)    
                        total += label_batch.size(0)
                        correct += (predicted == label_batch).sum().item()
                    
                    accuracy = 100.0 * correct / total
                    result = '[Test][Epoch: {0:d}, Accu: {1:.3f}%'.format(
                        epoch + 1, accuracy
                    )
                    print(result)
                    print()
                    if accuracy > best_accracy | (epoch+1) % 5 == 0:
                        print('Saving model in epoch: {0:d}'.format(epoch))
                        torch.save(model_net.state_dict(), '{0:s}/model_{1:03d}_{2:.3f}'.format(args.model, epoch+1, acc_f*100))
                        best_accracy = accuracy
                    acc_f.write(result)
                    acc_f.write('\n')
                    acc_f.flush()
                    # Testing finished, mark down the loss and accuray of the whole epoch
                    test_accuracy_list.append(100 * correct / len(test_dataloder.dataset))
                
                
            # Epoch finished.
            draw_acc_loss(epochs, train_acc=train_accuracy_list, train_loss=loss_list, test_acc=test_accuracy_list)
                
                
            print('Saving Final model in epoch')
            torch.save(model_net.state_dict(), '{0:s}/model_final'.format(args.model))
            print("Training Finish")


def main():
    if not os.path.exists(args.data):
        print('No dataset scanned!')
        return
    if not os.path.exists(args.model):
        os.makedirs(args.model)
    
    print('Start Training\n')
    train(model, int(EPOCH), train_dataloder, test_dataloder)


if __name__ == '__main__':
    main()