'''
@Author: your name
@Date: 2020-04-08 14:13:17
@LastEditTime: 2020-04-08 23:01:22
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \Pattern\torch_util.py
'''

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image

import os

class CorelDataset(Dataset):
    '''
    Description:
    ------------
    Custom implementation of pytorch's DataSet class
    '''    
    def __init__(self, root_dir, names_file, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.names_file = names_file
        self.trasforms = transform
        self.size = 0
        self.names_list = []

        self.transforms = transform
        if transform is None:
            self.transforms = transforms.Compose([ transforms.Resize((224,224))
                                                  ,transforms.ToTensor()])

        if not os.path.isfile(self.names_file):
            print('Name file does not exist, use util.make_namefile_file to make it')
        with open(self.names_file, 'r') as file:
            for line in file:
                self.names_list.append(line)
                self.size += 1
                
    
    def __getitem__(self, index):
        image_path, image_label = self.names_list[index].split(' ')
        image_label = int(image_label)
        if not os.path.isfile(image_path):
            print('No such image : ' + image_path)
            return None
        img = Image.open(image_path)
#         img = io.imread(image_path)
    
        img = self.transforms(img)
        
        return {'image':img, 'label':image_label}
        
        
    def __len__(self):
        return self.size

# *unit test for dataset
def dataset_test():
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    train_dataset = CorelDataset('./dataset/train',
                             './dataset/train/train.txt')
    # show images to make sure it works
    plt.figure()
    for cnt, i in enumerate(train_dataset):
        image = i['image']
        label = i['label']
        image = image.numpy().transpose(1, 2, 0)
        ax = plt.subplot(4, 4, cnt+1)
        ax.axis('off')
        ax.imshow(image)
        ax.set_title('label {},{}'.format(label, image.shape))
        if cnt == 15:
            break
    plt.show()
    train_dataloder = DataLoader(train_dataset,
                                batch_size=4,
                                shuffle=True,
                                num_workers=4)
    plt.figure()
    for batch_cnt, batch_i in enumerate(train_dataloder):
        images_batch, labels_batch = batch_i['image'], batch_i['label']
        grid = make_grid(images_batch)
        plt.imshow(grid.numpy().transpose(1, 2, 0)) 
        plt.axis('off')
        plt.show()
    plt.show()

# # ------------------------------------------------------------------------------
# from datetime import datetime
# import visdom
# class Visualizations:
#     """
#     Pytorch module visualization using visdom
#     """
#     def __init__(self, env_name=None):
#         if env_name is None:
#             env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
#         self.env_name = env_name
#         self.vis = visdom.Visdom(env=self.env_name)
#         self.loss_win = None

#     def plot_loss(self, loss, step):
#         self.loss_win = self.vis.line(
#             [loss],
#             [step],
#             win=self.loss_win,
#             update='append' if self.loss_win else None,
#             opts=dict(
#                 xlabel='Step',
#                 ylabel='Loss',
#                 title='Loss (mean per 10 steps)',
#             )
#         )

def accuracy_calc(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def get(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return str(entries)
        
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
    
if __name__ == "__main__":
    dataset_test()