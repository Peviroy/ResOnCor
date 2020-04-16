import os
import numpy as np
import cv2
from numpy import ndarray
import matplotlib.pyplot as plt

""" Generic API
"""


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
        image_paths = [path.path for path in it if (
            path.name.endswith('.jpg') or path.name.endswith('.png'))]
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
    iter_all_images = (cv2.resize(cv2.imread(image_path), image_size)
                       for image_path in image_paths)
    for i, image in enumerate(iter_all_images):
        if i == 0:
            all_images = np.empty(
                (len(image),) + image.shape, dtype=image.dtype)
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
    if 'train' in root or 'test' in root or 'val' in root:  # 递归
        for sub_dir in root:
            make_namefile_file(os.path.join(root_dir, sub_dir))
    else:
        index_file_name = os.path.split(root_dir)[-1] + '.txt'  # train.txt
        # make the index_file's path
        index_file = os.path.join(root_dir, index_file_name)
        if os.path.exists(index_file):
            os.remove(index_file)
        with open(index_file, 'w') as file_handle:  # create file
            i = -1
            for class_dir in root:  # search every subdir
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
    if 'train' in root or 'test' in root or 'val' in root:  # 递归
        for sub_dir in root:
            dir_name = str(sub_dir)
            sub_dir_list = count_pic_num(os.path.join(root_dir, sub_dir))
            list.append(dict({dir_name: sub_dir_list}))

    else:
        for class_dir in root:  # search every subdir
            if not os.path.isdir(os.path.join(root_dir, class_dir)):
                continue
            class_root = os.path.join(root_dir, class_dir)
            class_name = str(class_dir)
            class_count = 0
            with os.scandir(class_root) as it:  # eg. 'dataset/train/beach/xxx.jpg beach'
                for path in it:
                    if path.name.endswith('.jpg') or path.name.endswith('.png'):
                        class_count += 1
            list.append(dict({class_name: class_count}))
    return list


def draw_acc_loss(START_EPOCH: int, END_EPOCH: int, train_acc: list, train_loss: list, test_acc: list, savedir: str):
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
