import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
""" Useful api dedicated to pytorch
"""


# ! not used; dataloader shuffle seed
def worker_init_fn(worker_id):
    np.random.seed(7 + worker_id)


# ! not used;
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def accuracy_calc(output, target, topk=(1, )):
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


def adjust_learning_rate(optimizer, epoch, original_lr, decay_step):
    """Sets the learning rate to the initial LR decayed by 10 every step
    """
    lr = original_lr * (0.1 ** (epoch // decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# tring to un normalize the image so that a normal image can be displayed, however failed
def unnormalization(normalization, img_batch):
    mean, std = normalization
    unnor_img_batch = img_batch
    for i, channel in enumerate(unnor_img_batch):
        channel = channel * std[i] + mean[i]
    unnor_img_batch = unnor_img_batch.mul(255)
    return unnor_img_batch


def view_predicted(dataloader, model, device, normalization):
    """Display image input 、ground truth and predicted label
    """
    plt.figure()
    with torch.no_grad():
        for batch_cnt, batch_data in enumerate(dataloader):
            image_batch, label_batch = batch_data['image'], batch_data['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            # get predicted label
            outputs = model(image_batch)
            _, predicted = torch.max(outputs, 1)

            # draw
            grid = make_grid(image_batch)
            grid = unnormalization(normalization, grid)
            plt.imshow(grid.cpu().numpy().transpose(1, 2, 0))
            plt.axis('off')
            plt.show()
            # map the label number to class name
            groud_truth = label_batch.tolist()
            predicted_label = predicted.tolist()
            label_to_class = {
                0: 'beach',
                1: 'dinosaur',
                2: 'African',
                3: 'horse',
                4: 'bus',
                5: 'building',
                6: 'mountain',
                7: 'food',
                8: 'elephant',
                9: 'flower'
            }
            groud_truth = [label_to_class.get(x) for x in groud_truth]
            predicted_label = [label_to_class.get(x) for x in predicted_label]

            # list = []
            # for i in range(0, len(predicted)):
            #     dic = dict({'Predict: ': predicted[i], 'Groud truth: ': label_batch[i]})
            #     list.append(dic)

            print('Pridicted:', predicted_label)
            print('Groud truth:', groud_truth)

    plt.show()
