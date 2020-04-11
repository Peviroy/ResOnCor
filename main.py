import os
os.chdir(os.path.split(os.path.realpath(__file__))[0]) # make sure to change to the projict directory
import sys
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import argparse



parser = argparse.ArgumentParser(description="Resnet on CorelDataset")
parser.add_argument('--model', default='./model', help='folder to save models')
parser.add_argument('--data', default='./dataset', help='where the data set is stored')
parser.add_argument('--resnet_type', default='resnet18', help='kind of resnet used to train. Eg.resnet18, resnet50')
parser.add_argument('--batch', default=32, help='batch size of data input')
args = parser.parse_args()

# *Hpyter argument
EPOCH = 100
PRE_EPOCH = 0
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
                            

net = ResNet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
                


if __name__ == '__main__':
    if not os.path.exists(args.data):
        print('No dataset scanned!')
        pass

    if not os.path.exists(args.model):
        os.makedirs(args.model)
    
    print('Start Training-------------------------\n')
    best_accracy = 0
    with open('accuracy.txt', 'w') as acc_f:
        with open('log.txt', 'w') as log_f:
            for epoch in range(PRE_EPOCH, EPOCH):
                print('Epoch: {0:d} \n'.format(epoch+1))
                net.train()
                loss_sum = 0.0
                correct = 0.0
                total = 0.0

                print('Training..............')
                for batch_cnt, batch_data in enumerate(train_dataloder):
                    length = len(train_dataloder)
                    image_batch, label_batch = batch_data['image'], batch_data['label']
                    image_batch, label_batch = image_batch.to(device), label_batch.to(device)

                    optimizer.zero_grad()

                    outputs = net(image_batch)
                    loss = criterion(outputs, label_batch)
                    loss.backward()
                    optimizer.step()

                    loss_sum += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += label_batch.size(0)
                    correct += (predicted == label_batch).sum()

                    result = '[Train][Epoch:{0:d}, iter:{1:d}] Loss: {2:.03f} | Acc: {3:.3f}%%'.format(
                        epoch+1, (batch_cnt+1+epoch*length), loss_sum / (batch_cnt+1), correct / total * 100.0
                    )
                    print(result)
                    log_f.write(result)
                    log_f.write('\n')
                    log_f.flush()
                


                print('Testing..............')
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in test_dataloder:
                        net.eval()
                        image_batch, label_batch = batch_data['image'], batch_data['label']
                        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                        outputs = net(image_batch)

                        _, predicted = torch.max(outputs.data, 1)    
                        total += label_batch.size(0)
                        correct += (predicted == label_batch).sum()
                    
                    accuracy = 100.0 * correct / total
                    result = '[Test][Epoch: {0:d}, Accu: {1:.3f}%'.format(
                        epoch + 1, accuracy
                    )
                    print(result)
                    if accuracy > best_accracy:
                        if (epoch+1) > 50:
                            print('Saving model in epoch: {0:d}'.format(epoch))
                            torch.save(net.state_dict(), '{0:s}/model_{1:03d}'.format(args.model, epoch+1))
                        best_accracy = accuracy
                        acc_f.write('Better')
                    acc_f.write(result)
                    acc_f.write('\n')
                    acc_f.flush()

            print('Saving Final model in epoch')
            torch.save(net.state_dict(), '{0:s}/model_final'.format(args.model))
            print("Training Finish")
