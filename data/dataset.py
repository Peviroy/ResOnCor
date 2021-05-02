from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image

import os
"""Custom implementation of pytorch's DataSet class
"""


class CorelDataset(Dataset):
    '''
    Description:
    ------------
    Custom implementation of pytorch's DataSet class for Corel dataset
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
            self.transforms = transforms.Compose(
                [transforms.Resize((224, 224)),
                 transforms.ToTensor()])

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

        img = self.transforms(img)

        return {'image': img, 'label': image_label}

    def __len__(self):
        return self.size


# unit test for dataset
def dataset_test():
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    train_dataset = CorelDataset('./dataset/train', './dataset/train/train.txt')
    # show images to make sure it works
    plt.figure()
    for cnt, i in enumerate(train_dataset):
        image = i['image']
        label = i['label']
        image = image.numpy().transpose(1, 2, 0)
        ax = plt.subplot(4, 4, (cnt + 1))
        ax.axis('off')
        ax.imshow(image)
        ax.set_title('label {},{}'.format(label, image.shape))
        if cnt == 15:
            break
    plt.show()
    train_dataloder = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    plt.figure()
    for batch_cnt, batch_i in enumerate(train_dataloder):
        images_batch = batch_i['image']
        grid = make_grid(images_batch)
        plt.imshow(grid.numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.show()
    plt.show()


if __name__ == "__main__":
    dataset_test()
