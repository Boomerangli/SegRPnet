import os
import imageio
import numpy as np
import scipy.io as sio
import PIL
from PIL import Image
import torch
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import glob
import random
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.utils as v_utils


class VitnetTrainDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):

        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/image' % mode) + '/*.*'))
        # psth = os.path.join(root, '%s/image' % mode)
        # self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        self.files_Label = sorted(glob.glob(os.path.join(root, '%s/label' % mode) + '/*.*'))

        for i in range(len(self.files_A)):
            image_name = self.files_A[i].split("/")[-1]
            label_name = self.files_Label[i].split("/")[-1]
            if image_name != label_name:
                raise NameError("图片与标签不匹配")
        print("图片匹配成功...")

    def __getitem__(self, index):
        # zzq备注 对于彩色影像后加.convert('RGB')
        # label加.convert('L')，防止为二值影像时（'90'模式）高斯平滑会出错
        input_A = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        label = Image.open(self.files_Label[index % len(self.files_Label)]).convert('L')
        input_A = self.transform(input_A)
        label = self.transform(label)

        return input_A, label

    def __len__(self):
        return len(self.files_A)


class VitnetTestDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='test'):
        self.transform = transforms.Compose(transforms_)

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/image' % mode) + '/*.*'))
        root1 = os.path.join(root, '%s/image' % mode) + '/*.*'
        # self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        self.files_Label = sorted(glob.glob(os.path.join(root, '%s/label' % mode) + '/*.*'))

    def __getitem__(self, index):
        input_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        # input_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        # label,label_weight = self.transform(Image.open(self.files_Label[index % len(self.files_Label)]))
        label = Image.open(self.files_Label[index % len(self.files_Label)]).convert('L')

        label = self.transform(label)

        return input_A, label

    def __len__(self):
        return len(self.files_A)


class VitnetDataLoader:
    def __init__(self, config):
        self.config = config

        if config.data_mode == "imags":
            train_transforms_ = [
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]
            test_transforms_ = [
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ]

            train_dataset = VitnetTrainDataset(config.data_path, train_transforms_)
            test_dataset = VitnetTestDataset(config.data_path, test_transforms_)

            self.dataset_len = len(train_dataset)

            self.num_iterations = (self.dataset_len + config.train_batch_size - 1)

            self.train_loader = DataLoader(
                dataset=train_dataset,
                shuffle=False,
                num_workers=config.num_worker,
                batch_size=config.train_batch_size,
                drop_last=True
            )

            self.test_loader = DataLoader(
                dataset=test_dataset,
                shuffle=False,
                num_workers=config.num_worker,
                batch_size=config.test_batch_size,
                drop_last=True
            )

        elif config.data_mode == "numpy":
            raise NotImplementedError("This mode is not implemented YET")

        elif config.data_mode == "random":
            train_data = torch.randn(self.config.batch_size, self.config.input_channels, self.config.img_size,
                                     self.config.img_size)
            train_labels = torch.ones(self.config.batch_size).long()
            valid_data = train_data
            valid_labels = train_labels
            self.len_train_data = train_data.size()[0]
            self.len_valid_data = valid_data.size()[0]

            self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
            self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

            train = TensorDataset(train_data, train_labels)
            valid = TensorDataset(valid_data, valid_labels)

            self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
            self.test_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=False)

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def plot_samples_per_epoch(self, batch, epoch):
        """
        Plotting the batch images
        :param batch: Tensor of shape (B,C,H,W)
        :param epoch: the number of current epoch
        :return: img_epoch: which will contain the image of this epoch
        """
        img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
        v_utils.save_image(batch,
                           img_epoch,
                           nrow=4,
                           padding=2,
                           normalize=True)
        return imageio.imread(img_epoch)

    def make_gif(self, epochs):
        """
        Make a gif from a multiple images of epochs
        :param epochs: num_epochs till now
        :return:
        """
        gen_image_plots = []
        for epoch in range(epochs + 1):
            img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
            try:
                gen_image_plots.append(imageio.imread(img_epoch))
            except OSError as e:
                pass

        imageio.mimsave(self.config.out_dir + 'animation_epochs_{:d}.gif'.format(epochs), gen_image_plots, fps=2)

    def finalize(self):
        pass