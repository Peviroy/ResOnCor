from .corel import CorelDataset
from .voc import VOCDataset, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT
import torch
import cv2
import numpy as np

train_cfg = {
    'lr_epoch': (90, 120),
    'max_epoch': 150,
    'min_dim': {
        'yolo': [416, 416],
        'fcos': [320, 320]
    }
}


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


class YOLOBaseTransform:
    def __init__(self, size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def transform(self, image, size, mean, std):
        x = cv2.resize(image, (size, size)).astype(np.float32)
        x /= 255.
        x -= mean
        x /= std
        return x

    def __call__(self, image, boxes=None, labels=None):
        return self.transform(image, self.size, self.mean, self.std), boxes, labels


class FCOSBaseTransform:
    def __init__(self, size, mean=(104, 117, 123)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def transform(self, image, size, mean):
        x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x

    def __call__(self, image, boxes=None, labels=None):
        return self.transform(image, self.size, self.mean), boxes, labels
