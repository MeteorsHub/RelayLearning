import csv
import itertools
import os
from collections import OrderedDict

import PIL.Image
import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import Dataset

from datasets.basic import MultiSiteDataset
from utils.lung_cropping_tools import get_crop_model, learning_based_lung_cropping
from utils.project_utils import get_platform_specific_value
from utils.visualization import PALLATE


class MediastinumTumorDataset(MultiSiteDataset, Dataset):
    def __init__(self, train: bool, task: str, num_classes: int, task_id=None, task_name=None,
                 root_path=None, img_folder=None, train_ratio=0.8, num_test_samples=None,
                 hu_min=-160,  # 0 in png img represent this hu value
                 hu_window=(-160, 240), out_range=(0.0, 1.0), img_size=(256, 256),
                 add_bg_label=False,  # if true, additional bg label (hu less than -500) is added
                 add_bone_label=False,  # if true, additional bone label (hu larger than 800) is added
                 num_replica=1,  # repeat multiple epochs in one epoch
                 study_wise=True,  # if true, train val split will be study_wise. Otherwise, slice_wise
                 neg_pos_ratio=None,  # if not None, random select this number of negative imgs every one positive img
                 ):
        super().__init__(train, task, num_classes, task_id, task_name)
        assert train_ratio is None or num_test_samples is None

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

        self.img_size = img_size
        self.hu_window = hu_window
        self.hu_min = hu_min
        self.out_range = out_range
        self.num_replica = num_replica if train else 1
        self.study_wise = study_wise
        self.add_bg_label = add_bg_label
        self.add_bone_label = add_bone_label
        self.neg_pos_ratio = neg_pos_ratio if self.train else None
        if self.neg_pos_ratio is not None:
            assert isinstance(self.neg_pos_ratio, int) and self.neg_pos_ratio > 1

        instances = list()
        for path in self.img_paths:
            if not os.path.exists(path):
                self.logger.critical('Dataset path %s do not exist' % path)
                exit(-1)
            with open(os.path.join(path, 'meta.txt')) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row['img_file'] = os.path.join(path, row['img_file'])
                    row['label_file'] = os.path.join(path, row['label_file'])
                    row['positive'] = row['positive'] in ['True', 'true', 'TRUE', 't', 'T']
                    instances.append(row)
        study_instances = OrderedDict()
        for ins in instances:
            study_id = ins['study_id']
            if study_id in study_instances:
                study_instances[study_id].append(ins)
            else:
                study_instances[study_id] = [ins]

        # split train/test
        rng = np.random.default_rng(seed=1234)
        # this rng is to confirm every epoch and experiment share the same train val split.
        # you should use np.random.xxx to shuffle sample order in different epochs
        self.rng = rng
        if not self.study_wise:
            self.rng.shuffle(instances)
            if train_ratio is not None:
                if self.train:
                    img_start, img_end = 0, int(len(instances) * train_ratio)
                else:
                    img_start, img_end = int(len(instances) * train_ratio), len(instances)
            else:
                assert num_test_samples < len(instances)
                if self.train:
                    img_start, img_end = 0, len(instances) - num_test_samples
                else:
                    img_start, img_end = len(instances) - num_test_samples, len(instances)
            instances = instances[img_start:img_end]
        else:
            items = list(study_instances.items())
            self.rng.shuffle(items)
            study_instances_list = OrderedDict(list(items))
            if train_ratio is not None:
                if self.train:
                    study_start, study_end = 0, int(len(study_instances_list) * train_ratio)
                else:
                    study_start, study_end = int(len(study_instances_list) * train_ratio), len(study_instances_list)
            else:
                assert num_test_samples < len(study_instances_list)
                if self.train:
                    study_start, study_end = 0, len(study_instances_list) - num_test_samples
                else:
                    study_start, study_end = len(study_instances_list) - num_test_samples, len(study_instances_list)
            self.study_ids = list(study_instances.keys())[study_start:study_end]
            instances = []
            for k in self.study_ids:
                instances += study_instances[k]

        self.num_samples = len(instances)
        if self.neg_pos_ratio is not None:
            label_ps = [ins['positive'] for ins in instances]
            indexes = list(range(len(label_ps)))
            positive_indexes = [ind for ind in indexes if label_ps[ind]]
            negative_indexes = [ind for ind in indexes if not label_ps[ind]]
            if len(negative_indexes) / len(positive_indexes) > self.neg_pos_ratio:
                num_selected_negs = self.neg_pos_ratio * len(positive_indexes)
                negative_indexes = \
                    np.random.choice(negative_indexes, num_selected_negs, replace=False).tolist()
            else:
                num_more_negs = self.neg_pos_ratio * len(positive_indexes) - len(negative_indexes)
                negative_indexes += np.random.choice(negative_indexes, num_more_negs, replace=True).tolist()
        self.ori_instances = instances
        self.instances = list()
        for _ in range(self.num_replica):
            if self.neg_pos_ratio is not None:
                pos_indexes, neg_indexes = positive_indexes.copy(), negative_indexes.copy()
                if train:
                    np.random.shuffle(pos_indexes)
                    np.random.shuffle(neg_indexes)
                tuples = [[pos_indexes[i]] + neg_indexes[i * self.neg_pos_ratio:(i + 1) * self.neg_pos_ratio]
                          for i in range(len(pos_indexes))]
                final_indexes = list(itertools.chain(*tuples))
            else:
                final_indexes = list(range(len(instances)))
                if train:
                    np.random.shuffle(final_indexes)
            inst = [instances[i] for i in final_indexes]
            self.instances += inst

        def _normalize(img: torch.Tensor):
            img = img.type(torch.float)  # real hu value
            img = (img - self.hu_window[0]) / (self.hu_window[1] - self.hu_window[0])  # hu_window to 0~1
            img = img.clip(0, 1)  # img clip to 0~1
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

    def __getitem__(self, item):
        ins = self.instances[item]
        img_file, label_file = ins['img_file'], ins['label_file']
        img, label = self.load_img(img_file, label_file)
        img, label = img.astype(np.float32), label.astype(np.int32)
        img = img + self.hu_min  # transform to real hu value

        if self.task == 'segmentation':
            if self.add_bone_label:
                label = np.where(img > 150, np.zeros_like(label), label + 1)
            if self.add_bg_label:
                label = np.where(img < -500, np.zeros_like(label), label + 1)

        img = self.img_transforms(img)
        if self.task == 'classification':
            label = label.astype(np.int64)
        else:
            label = self.label_transforms(label).squeeze(0)
        return img, label

    def __len__(self):
        return len(self.instances)

    def get_subject_item_id(self, item: int):
        subject_id = self.instances[item]['study_id']
        slice_id = int(self.instances[item]['instance_id'].replace(subject_id + '_', ''))
        return subject_id, slice_id

    def load_img(self, img_file, label_file):
        img = np.array(PIL.Image.open(img_file)).astype(np.int16)
        if self.task == 'segmentation':
            label = np.array(PIL.Image.open(label_file)).astype(np.int8)
        elif self.task == 'classification':
            label = np.array(self.task_id - 1).astype(np.int8)
        else:
            label = np.array(PIL.Image.open(label_file)).astype(np.int8)
        return img, label


def transfer_mediastinum_tumor(src_path, dst_path, institution_map, crop=False, crop_square=True,
                               hu_window=(-160, 240), norm_uint8=False, save_label=True,
                               positive_threshold=None, num_slices_threshold=None, num_studies=None):
    """ Build mediastinum_tumor dataset from 3d ndarray to png slice format"""
    assert (positive_threshold is None) != (num_slices_threshold is None)

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    summary_file = os.path.join(dst_path, 'meta.txt')
    with open(summary_file, 'w') as f:
        for i_institution, (ins_name, ins_folders) in enumerate(institution_map.items(), 1):
            print('begin to process %d/%d institutions' % (i_institution, len(institution_map)))
            instances = []
            for ins_folder in ins_folders:
                instances += [os.path.join(src_path, ins_folder, ins)
                              for ins in os.listdir(os.path.join(src_path, ins_folder))]
            if num_studies is not None:
                if num_studies > len(instances):
                    print('warning: require %d studies but only %s studies got' % (num_studies, len(instances)))
                else:
                    import random
                    random.seed(1)
                    instances = random.sample(instances, num_studies)

            if not os.path.exists(os.path.join(dst_path, ins_name)):
                os.makedirs(os.path.join(dst_path, ins_name))
            if crop:
                crop_model = get_crop_model()

            total_num_valid_slices = 0
            total_num_valid_instances = 0
            for i_ins, ins_file in enumerate(instances, 1):
                ins_data = np.load(ins_file)
                # img_shape = [h, w, d]. label_shape: [d, h, w].
                image = ins_data['data'].transpose(2, 0, 1).astype(np.int32)
                label_map = ins_data['labels'].astype(np.int8)
                label_map = (label_map > 0).astype(np.int8)
                if label_map.shape != image.shape:
                    print('warning: [%s] image and label have different shape: %s and %s'
                          % (ins_file, image.shape, label_map.shape))
                    continue
                # final shape [d, h, w]
                if label_map.max() == 0:
                    print('warning: instance %s has no positive labels' % ins_file)

                if crop:
                    dl, dh, hl, hh, wl, wh = learning_based_lung_cropping(image.transpose(1, 2, 0), model=crop_model)
                    if hh - hl + 1 < 0.2 * image.shape[1]:
                        print('warning: ins %s cropped h is only %d, while original is %d'
                              % (ins_file, hh - hl + 1, image.shape[1]))
                    if wh - wl + 1 < 0.2 * image.shape[1]:
                        print('warning: ins %s cropped w is only %d, while original is %d'
                              % (ins_file, wh - wl + 1, image.shape[2]))
                    ratio = (hh - hl + 1) / (wh - wl + 1)
                    if ratio > 1.0 or ratio < 0.3:
                        print('warning: ins %s cropped ratio is %1.2f' % (ins_file, ratio))
                    if crop_square:
                        h_pad = image.shape[1] // 2
                        image = np.pad(image, ((0, 0), (h_pad, h_pad), (0, 0)), mode='edge')
                        label_map = np.pad(label_map, ((0, 0), (h_pad, h_pad), (0, 0)), mode='edge')
                        h_need_more = (wh - wl + 1) - (hh - hl + 1)
                        hl = hl + h_pad - h_need_more // 2
                        hh = hl + (wh - wl)
                    image = image[dl:dh + 1, hl:hh + 1, wl:wh + 1]
                    label_map = label_map[dl:dh + 1, hl:hh + 1, wl:wh + 1]

                label_on_depth = label_map.sum(axis=(1, 2)).astype(np.float32)
                sorted_index = np.flip(np.argsort(label_on_depth))
                image = image[sorted_index]
                label_map = label_map[sorted_index]
                label_on_depth = label_on_depth[sorted_index]
                if positive_threshold is not None:
                    if isinstance(positive_threshold, float) and 0. < positive_threshold <= 1.:
                        spatial_size = label_map.shape[1] * label_map.shape[2]
                        ratio_label_on_depth = label_on_depth / spatial_size
                        num_valid_slices = (ratio_label_on_depth > positive_threshold).astype(np.int8).sum()
                    elif isinstance(positive_threshold, int):
                        num_valid_slices = (label_on_depth > positive_threshold).astype(np.int).sum()
                    else:
                        raise AttributeError
                elif num_slices_threshold is not None:
                    if isinstance(num_slices_threshold, float) and 0. < num_slices_threshold <= 1.:
                        num_valid_slices = int(len(label_on_depth) * num_slices_threshold)
                    elif isinstance(num_slices_threshold, int):
                        if num_slices_threshold > len(label_on_depth):
                            # print(
                            #     'warning: num_slices is larger than total slices and will be set to total slices in %s' % ins_file)
                            num_valid_slices = len(label_on_depth)
                        else:
                            num_valid_slices = num_slices_threshold
                    else:
                        raise AttributeError
                else:
                    raise AttributeError

                total_num_valid_slices += num_valid_slices
                if num_valid_slices > 0:
                    total_num_valid_instances += 1
                if num_valid_slices == 0:
                    print('warning: (ins %d/%d) %d/%d valid slices in instance %s' % (
                        i_ins, len(instances), num_valid_slices, len(label_on_depth), ins_file))
                else:
                    print('(ins %d/%d) %d/%d valid slices in instance %s' % (
                        i_ins, len(instances), num_valid_slices, len(label_on_depth), ins_file))
                for slice_id in range(num_valid_slices):
                    img, label = image[slice_id], label_map[slice_id]
                    # if label.max() == 0:
                    #     print('warning: slice %d in instance %s has no positive labels' % (slice_id, ins_file))
                    img = (np.clip(img, hu_window[0], hu_window[1]) - hu_window[0]).astype(np.uint16)
                    if norm_uint8:
                        img = (255 * img.astype(np.float32) / (hu_window[1] - hu_window[0])).astype(np.uint8)
                        img = PIL.Image.fromarray(img, mode='L')
                    else:
                        img = PIL.Image.fromarray(img, mode='I;16')
                    label = PIL.Image.fromarray(label, mode='P')
                    label.putpalette(PALLATE)
                    img_tgt_file = os.path.join(dst_path, ins_name,
                                                os.path.basename(ins_file).replace('.npz',
                                                                                   '_%03d.png' % (slice_id + 1)))
                    label_tgt_file = img_tgt_file.replace('.png', '_label.png')
                    img.save(img_tgt_file)
                    if save_label:
                        label.save(label_tgt_file)
            print(
                'institution %s; num_instance: %d; num slices %d'
                % (ins_name, total_num_valid_instances, total_num_valid_slices))
            meta_analyse(os.path.join(dst_path, ins_name))
            f.write('institution: %s\n\tnum_instances: %d\tnum_slices: %d\n'
                    % (ins_name, total_num_valid_instances, total_num_valid_slices))


def meta_analyse(folder):
    """ build meta file for the dataset """
    assert os.path.exists(folder)
    img_files = [ins for ins in os.listdir(folder) if ins.endswith('.png') and '_label' not in ins]
    field_names = ['instance_id', 'study_id', 'img_file', 'label_file', 'positive']
    with open(os.path.join(folder, 'meta.txt'), 'w') as f:
        writer = csv.DictWriter(f, field_names)
        writer.writeheader()
        for img_f in img_files:
            study_id = '_'.join(img_f.split('_')[:-1])
            img_id = img_f.replace('.png', '')
            label_f = img_f.replace('.png', '_label.png')
            label = np.array(PIL.Image.open(os.path.join(folder, label_f))).astype(np.int8)
            positive = np.any(label)
            writer.writerow({
                'instance_id': img_id,
                'study_id': study_id,
                'img_file': img_f,
                'label_file': label_f,
                'positive': positive
            })
