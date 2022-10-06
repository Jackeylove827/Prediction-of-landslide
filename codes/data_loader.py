#coding='utf-8'
import os
import torch
import torch.utils.data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
from PIL import Image
import torchvision
import skimage.measure as skm
from pdb import set_trace as stc

class my_dataset(torch.utils.data.Dataset):
    def __init__(self, cfg,  data_list, label_root, data_root, data_flag='Test'):
        super(my_dataset, self).__init__()
        self.data_root = data_root
        self.label_root = label_root
        self.cfg = cfg
        self.data_flag = data_flag

        data_file = open(data_list, 'r', encoding='utf-8')
        data_lines = data_file.readlines()

        self.img_datas  = []
        self.img_labels = []
        for line in data_lines:
            line = line.strip()
            if len(line.split(',')) > 1: # 一行两个记录，左边为图像，右边为标签
                img_name = line.split(',')[0]
                label_name = line.split(',')[1]
                self.img_datas.append(img_name)
                self.img_labels.append(label_name)
            else:
                raise RuntimeError

        data_file.close()

        if self.data_flag == 'Test':
            if self.cfg.test.test_mode == 1: # 有标签，中心裁剪
                self.transform = transforms.Compose(
                    [
                        transforms.CenterCrop(self.cfg.train.crop_size),
                        transforms.Resize(self.cfg.train.resize),
                        transforms.ToTensor(),
                    ]
                )
            elif self.cfg.test.test_mode == 0: # 无标签，全图resize
                self.transform = transforms.Compose(
                    [
                        transforms.Resize(self.cfg.train.resize),
                        transforms.ToTensor(),
                    ]
                )
        elif self.data_flag == 'Train':
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(self.cfg.train.crop_size),
                    transforms.Resize(self.cfg.train.resize),
                    transforms.ToTensor(),
                ]
            )

    def __getitem__(self, index):

        img_path = self.img_datas[index]
        img_path = img_path.replace(os.path.dirname(img_path), '').replace('/', '')
        label_path = self.img_labels[index]
        label_path = label_path.replace(os.path.dirname(label_path), '').replace('/', '')

        img = Image.open(os.path.join(self.data_root, img_path)).convert('L') #
        if self.cfg.test.test_mode == 1: # 有标签
            label = Image.open(os.path.join(self.label_root, label_path)).convert('L') #
        elif self.cfg.test.test_mode == 0: # 无标签，用于测试
            label = np.zeros(img.size).reshape(img.size[::-1])
            label = Image.fromarray(label)

        img = self.transform(img) * 255 - 128
        label = self.transform(label)

        return img, label, self.img_datas[index]

    def __len__(self):
        return len(self.img_datas)

def get_dataloader(cfg, data_flag = ''):

    if data_flag == 'Train':
        dataset = my_dataset(cfg,  cfg.data.train_list, cfg.data.train_label_root, cfg.data.train_root, data_flag)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=cfg.train.batch_size,
                                                shuffle=True,
                                                num_workers=cfg.train.num_workers,
                                                drop_last=False,
                                                pin_memory=True)
    elif data_flag == 'Test':
        dataset = my_dataset(cfg,  cfg.data.test_list, cfg.data.test_label_root, cfg.data.test_root, data_flag)
        test_batch = 1
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=test_batch,
                                                shuffle=False,
                                                num_workers=cfg.train.num_workers,
                                                drop_last=False,
                                                pin_memory=True)
    else:
        raise 'Error happens in dataloader ... '

    return dataloader




