import os

import PIL
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from datasets.basic import MultiSiteDataset
from utils.project_utils import get_platform_specific_value


class FundusDataset(MultiSiteDataset, Dataset):
    def __init__(self, train: bool, task: str = 'segmentation', num_classes: int = 3, task_id=None, task_name=None,
                 root_path=None, img_folder=None, use_roi=True, img_size=(128, 128), out_range=(-1, 1), num_replica=1):
        super().__init__(train, task, num_classes, task_id, task_name)
        if img_folder is None:
            self.logger.critical('Dataset path %s is None')
            exit(-1)
        if not (isinstance(img_folder, list) or isinstance(img_folder, tuple)):
            img_folder = [img_folder]
        if root_path is not None:
            root_path = get_platform_specific_value(root_path)
            self.img_paths = [os.path.join(root_path, folder) for folder in img_folder]
        else:
            self.img_paths = img_folder

        self.use_roi = use_roi
        self.img_size = img_size
        self.out_range = out_range
        self.num_replica = num_replica if train else 1

        all_img_files = list()
        for img_path in self.img_paths:
            img_path = os.path.join(img_path, 'train' if self.train else 'test')
            if self.use_roi:
                img_path = os.path.join(img_path, 'ROIs')
            img_files = [f for f in os.listdir(os.path.join(img_path, 'image'))
                         if f.endswith('.png') and not f.startswith('.')]
            label_files = [f for f in os.listdir(os.path.join(img_path, 'mask'))
                           if f.endswith('.png') and not f.startswith('.')]
            assert set(img_files) == set(label_files)
            assert len(img_files) == len(label_files) and len(img_files) == len(set(img_files))
            all_img_files += [os.path.join(img_path, 'image', img_f) for img_f in img_files]

        self.num_samples = len(all_img_files)
        self.img_files = list()
        for _ in range(self.num_replica):
            _all_img_files = all_img_files.copy()
            if train:
                np.random.shuffle(_all_img_files)
            self.img_files += _all_img_files
        self.label_files = [img_f.replace('/image/', '/mask/') for img_f in self.img_files]

        def _normalize(img: torch.Tensor):
            img = img / 255.  # 0~255 to 0~1
            img = img * (self.out_range[1] - self.out_range[0]) + self.out_range[0]
            return img

        self.img_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(_normalize),
            torchvision.transforms.Resize(self.img_size, torchvision.transforms.InterpolationMode.BILINEAR)
        ])
        self.label_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.img_size, torchvision.transforms.InterpolationMode.NEAREST)
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        img_f = self.img_files[item]
        label_f = self.label_files[item]
        img, label = self.load_img(img_f, label_f)
        img = self.img_transforms(img.astype(np.float32))
        label = self.label_transforms(label.astype(np.int32)).squeeze(0)
        return img, label

    def load_img(self, img_file, label_file):
        img = np.array(PIL.Image.open(img_file)).astype(np.uint8)
        label = np.array(PIL.Image.open(label_file)).astype(np.float32)
        label = 255 - label.mean(axis=2)
        label = np.round(2 * label / 255).astype(np.int8)
        return img, label

    def get_subject_item_id(self, item: int):
        return item, item
