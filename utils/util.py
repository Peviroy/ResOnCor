import os
import numpy as np
import cv2
from numpy import ndarray
import matplotlib.pyplot as plt
import torch
""" Generic API
"""


def generate_dxdywh(gt_label, w, h, s):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # compute the center, width and height
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    if box_w < 1e-4 or box_h < 1e-4:
        # print('A dirty data !!!')
        return False

    # map center point of box to the grid cell
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # compute the (x, y, w, h) for the corresponding grid cell
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w)
    th = np.log(box_h)
    weight = 2.0 - (box_w / w) * (box_h / h)

    return grid_x, grid_y, tx, ty, tw, th, weight


def gt_creator(input_size, stride, label_lists=[]):
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    w = input_size
    h = input_size

    # We  make gt labels by anchor-free method and anchor-based method.
    ws = w // stride
    hs = h // stride
    s = stride
    gt_tensor = np.zeros([batch_size, hs, ws, 1 + 1 + 4 + 1])

    # generate gt whose style is yolo-v1
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            gt_class = int(gt_label[-1])
            result = generate_dxdywh(gt_label, w, h, s)
            if result:
                grid_x, grid_y, tx, ty, tw, th, weight = result

                if grid_x < gt_tensor.shape[2] and grid_y < gt_tensor.shape[1]:
                    gt_tensor[batch_index, grid_y, grid_x, 0] = 1.0
                    gt_tensor[batch_index, grid_y, grid_x, 1] = gt_class
                    gt_tensor[batch_index, grid_y, grid_x, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[batch_index, grid_y, grid_x, 6] = weight

    gt_tensor = gt_tensor.reshape(batch_size, -1, 1 + 1 + 4 + 1)

    return torch.from_numpy(gt_tensor).float()


def compute_iou(pred_box, gt_box):
    # calculate IoU
    # [l, t, r, b]

    w_gt = gt_box[:, :, 0] + gt_box[:, :, 2]
    h_gt = gt_box[:, :, 1] + gt_box[:, :, 3]
    w_pred = pred_box[:, :, 0] + pred_box[:, :, 2]
    h_pred = pred_box[:, :, 1] + pred_box[:, :, 3]
    S_gt = w_gt * h_gt
    S_pred = w_pred * h_pred
    I_h = torch.min(gt_box[:, :, 1], pred_box[:, :, 1]) + torch.min(gt_box[:, :, 3], pred_box[:, :,
                                                                                              3])
    I_w = torch.min(gt_box[:, :, 0], pred_box[:, :, 0]) + torch.min(gt_box[:, :, 2], pred_box[:, :,
                                                                                              2])
    S_I = I_h * I_w
    U = S_gt + S_pred - S_I + 1e-20
    IoU = S_I / U

    return IoU


def get_image_paths(directory: str) -> list:
    '''
    Description: 
    ------------
        Get the path of images 
    Param:
    ------------
        directory: {String}, directory of the images 
    Return: 
    ------------
        list containing images' paths
    '''
    with os.scandir(directory) as it:
        image_paths = [
            path.path for path in it if (path.name.endswith('.jpg') or path.name.endswith('.png'))
        ]
    return image_paths


def load_images(image_paths: list, image_size=(256, 256)) -> ndarray:
    '''
    Description: 
    ------------
        Get images according to the path list in the form of numpy array
    Param:
    ------------
        image_path: {String}, list containing images' paths
    Return: 
    ------------
        list of images in the form of numpy array
    '''
    iter_all_images = (cv2.resize(cv2.imread(image_path), image_size) for image_path in image_paths)
    image = iter_all_images[0]
    all_images = np.empty((len(image), ) + image.shape, dtype=image.dtype)
    for i, image in enumerate(iter_all_images):
        all_images[i] = image
    return all_images


def make_namefile_file(root_dir: str):
    """
    Description: 
    ------------
        Make a dataset index(Format: img_path, label)
    Param:
    ------------
        root_dir: {String}, root directory of images;
        eg: root_dir is /path/to/dataset/ 
        dataset/
            ├── test
            │   ├── African people
            │   ├── beach
            │   ├── building
            │   ├── bus
            │   ├── dinosaur
            │   ├── elephant
            │   ├── flower
            │   ├── food
            │   ├── horse
            │   └── mountain
            └── train
                ├── African people
                ├── beach
                ├── building
                ├── bus
                ├── dinosaur
                ├── elephant
                ├── flower
                ├── food
                ├── horse
                └── mountain
    Return: 
    ------------
        None
    """
    root = os.listdir(root_dir)
    if 'train' in root or 'test' in root or 'val' in root: # 递归
        for sub_dir in root:
            make_namefile_file(os.path.join(root_dir, sub_dir))
    else:
        index_file_name = os.path.split(root_dir)[-1] + '.txt' # train.txt
        # make the index_file's path
        index_file = os.path.join(root_dir, index_file_name)
        if os.path.exists(index_file):
            os.remove(index_file)
        with open(index_file, 'w') as file_handle: # create file
            i = -1
            for class_dir in root: # search every subdir
                i += 1
                class_root = os.path.join(root_dir, class_dir)
                if not os.path.isdir(class_root):
                    i -= 1
                    continue
                # eg. 'dataset/train/beach/xxx.jpg beach'
                with os.scandir(class_root) as it:
                    for path in it:
                        if path.name.endswith('.jpg') or path.name.endswith('.png'):
                            path_and_label = path.path + ' ' + str(i)
                            file_handle.write(path_and_label)
                            file_handle.write('\n')


def count_pic_num(root_dir: str):
    list = []
    root = os.listdir(root_dir)
    if 'train' in root or 'test' in root or 'val' in root: # 递归
        for sub_dir in root:
            dir_name = str(sub_dir)
            sub_dir_list = count_pic_num(os.path.join(root_dir, sub_dir))
            list.append(dict({dir_name: sub_dir_list}))

    else:
        for class_dir in root: # search every subdir
            if not os.path.isdir(os.path.join(root_dir, class_dir)):
                continue
            class_root = os.path.join(root_dir, class_dir)
            class_name = str(class_dir)
            class_count = 0
            with os.scandir(class_root) as it: # eg. 'dataset/train/beach/xxx.jpg beach'
                for path in it:
                    if path.name.endswith('.jpg') or path.name.endswith('.png'):
                        class_count += 1
            list.append(dict({class_name: class_count}))
    return list


def draw_acc_loss(START_EPOCH: int, END_EPOCH: int, train_acc: list, train_loss: list,
                  test_acc: list, savedir: str):
    """
    Description: 
    ------------
        Draw pie according to the acc and loss;

    Output:
    -------
        Two pictures; 
            One of them shows the accuracy of train and test; the other shows the loss of train

    Problems:
    ---------
        The output pictures is not fine grained
    """
    X_axis = range(START_EPOCH, END_EPOCH)
    Y_acc = [train_acc, test_acc]
    Y_loss = train_loss

    plt.figure()
    plt.plot(X_axis, Y_acc[0], '-', c='blue')
    plt.plot(X_axis, Y_acc[1], '-', c='green')
    plt.title('Accuracy vs. epoches')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'])
    plt.savefig(savedir + "accuracy.jpg")

    plt.figure()
    plt.plot(X_axis, Y_loss, '.-')
    plt.title('Train loss vs. epoches')
    plt.ylabel('Loss')
    plt.savefig(savedir + "loss.jpg")

    # plt.show()


if __name__ == "__main__":
    # root_dir = 'dataset'
    # make_namefile_file(root_dir)
    print(count_pic_num('./dataset'))
