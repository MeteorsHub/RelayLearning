import copy
import json
import os
from collections import OrderedDict

import PIL.Image
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from datasets.basic import MultiSiteDataset
from utils.project_utils import get_platform_specific_value


class MidlineDataset(MultiSiteDataset, Dataset):
    def __init__(self, train: bool, task: str = 'segmentation', num_classes: int = 2, task_id=None, task_name=None,
                 root_path=None, img_folder=None, train_ratio=0.8, img_size=(256, 256), out_range=(-1, 1),
                 pseudo_label=False,  # if true, add bg label
                 drop_negative=False,  # if true, drop negative samples that contain no brain midline
                 num_replica=1
                 ):
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

        self.train_ratio = train_ratio
        self.img_size = img_size
        self.out_range = out_range
        self.pseudo_label = pseudo_label
        self.drop_negative = drop_negative
        self.num_replica = num_replica if train else 1

        self.meta = self.determine_num_samples()

        studies = OrderedDict()
        for img_path in self.img_paths:
            folder_meta = self.meta[img_path]
            for ins, v in folder_meta.items():
                if self.drop_negative:
                    num_slices, slices = v['num_pos_slices'], v['pos_slices']
                else:
                    num_slices, slices = v['num_slices'], v['slices']
                if ins in studies:
                    print('error: dup ins_name')
                    exit(1)
                studies[ins] = {
                    'study_id': v['ins_id'],
                    'img_files': [os.path.join(img_path, ins, s + '.jpg') for s in slices],
                    'label_files': [os.path.join(img_path, ins, 'mask_' + s + '.png') for s in slices],
                    'num_slices': num_slices,
                }

        # split train/test
        rng = np.random.default_rng(seed=1234)
        # this rng is to confirm every epoch and experiment share the same train val split.
        # you should use np.random.xxx to shuffle sample order in different epochs
        self.rng = rng

        items = list(studies.items())
        self.rng.shuffle(items)
        if self.train_ratio is not None:
            if self.train:
                study_start, study_end = 0, int(len(studies) * self.train_ratio)
            else:
                study_start, study_end = int(len(studies) * self.train_ratio), len(studies)
            items = items[study_start:study_end]
        studies = OrderedDict(items)
        self.studies = studies

        self.num_samples = sum([study['num_slices'] for study in self.studies.values()])
        slices = list()
        for ins_name, ins_meta in self.studies.items():
            for i in range(ins_meta['num_slices']):
                slices.append({'instance': ins_name, 'instance_id': ins_meta['study_id'], 'slice_id': i})

        self.slices = list()
        for _ in range(self.num_replica):
            _slices = copy.deepcopy(slices)
            if train:
                np.random.shuffle(_slices)
            self.slices += _slices

        def _normalize(img: torch.Tensor):
            img = img / 255.
            img = img * (self.out_range[1] - self.out_range[0]) + self.out_range[0]  # 0~1 to out_range
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
        return len(self.slices)

    def __getitem__(self, item):
        item = self.slices[item]
        ins_name, slice_id = item['instance'], item['slice_id']
        img_file = self.studies[ins_name]['img_files'][slice_id]
        label_file = self.studies[ins_name]['label_files'][slice_id]
        img, label = self.load_img(img_file, label_file)

        img = self.img_transforms(img.astype(np.float32))
        label = self.label_transforms(label.astype(np.int32)).squeeze(0)
        return img, label

    def load_img(self, img_file, label_file):
        img = np.array(PIL.Image.open(img_file)).astype(np.uint8)
        label = np.array(PIL.Image.open(label_file)).astype(np.int16)

        if img.ndim == 3:
            img = img.mean(axis=2)
        # 0: bg; 1: clear_fg; 2: vague_fg; 11: unknown
        label = np.where(label > 10, np.zeros_like(label), label)
        label = (label > 0.5).astype(np.int8)

        if self.pseudo_label:
            bg_mask = np.logical_and(img < 10, label == 0)
            label = np.where(bg_mask, np.zeros_like(label), label + 1)
        return img, label

    def get_subject_item_id(self, item: int):
        return self.slices[item]['instance_id'], self.slices[item]['slice_id']

    def determine_num_samples(self):
        meta_file = 'meta.json'
        meta = dict()
        # {'img_path': {'ins_name': {'num_slices': int, 'num_pos_slices': int,
        #                            'slices': [file1, file2...], 'pos_slices': []}}}

        for img_path in self.img_paths:
            folder_meta = {}
            if os.path.exists(os.path.join(img_path, meta_file)):
                with open(os.path.join(img_path, meta_file)) as f:
                    folder_meta = json.load(f)
            else:
                instances = [f for f in os.listdir(img_path)
                             if not f.startswith('.') and os.path.isdir(os.path.join(img_path, f))]
                ins_ids = dict(zip(sorted(instances), list(range(len(instances)))))
                for ins in instances:
                    ins_name = ins
                    ins_id = ins_ids[ins]
                    slices = [f.replace('.jpg', '') for f in os.listdir(os.path.join(img_path, ins))
                              if f.endswith('.jpg') and not f.startswith('mask_') and not f.startswith('.')]
                    pos_slices = []
                    for slice in slices:
                        label = np.array(PIL.Image.open(
                            os.path.join(img_path, ins, 'mask_' + slice + '.png'))).astype(np.int16)
                        # 0: bg; 1: clear_fg; 2: vague_fg; 11: unknown
                        label = np.where(label > 10, np.zeros_like(label), label)
                        label = (label > 0.5).astype(np.int8)
                        if label.max() > 0:
                            pos_slices.append(slice)
                    folder_meta[ins_name] = {
                        'ins_id': ins_id,
                        'num_slices': len(slices), 'slices': slices,
                        'num_pos_slices': len(pos_slices), 'pos_slices': pos_slices,
                    }
                with open(os.path.join(img_path, meta_file), 'w') as f:
                    json.dump(folder_meta, f, indent=2)
            meta[img_path] = folder_meta
        return meta
