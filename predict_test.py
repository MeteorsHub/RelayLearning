import argparse
import csv
import datetime
import logging
import os

import SimpleITK as sitk
import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import get_dataset
from models import get_model
from strategies import get_strategy
from utils.metrics import Dice, DiceSampleWise, HausdorffDistance
from utils.project_utils import load_config, process_config, set_logger

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='fundus/relay', help='Which config is loaded from configs')
parser.add_argument('-g', '--gpus', type=str, default=None,
                    help='Gpu ids to use, like "1,2,3". If None, use cpu. '
                         'Note: in predict_test you should only use a single node')
parser.add_argument('-n', '--note', type=str, default='default',
                    help='Note to identify this experiment, like "first_version"... Should not contain space')
parser.add_argument('-s', '--save_results', action='store_true',
                    help='If set, save the prediction file to disk. Only support 3D medical images')
parser.add_argument('-v', '--verbose', action='store_true', help='If set, log debug level, else info level')
args = parser.parse_args()

if args.gpus is not None:
    assert len(args.gpus.split(',')) == 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
exp_path = os.path.join('./exps', args.config, args.note)


def predict_test():
    # torch.autograd.set_detect_anomaly(True)
    config_file = os.path.join('configs', args.config + '.yaml')
    ori_config = load_config(config_file)
    config = process_config(ori_config)
    time_str = datetime.datetime.now().strftime('%Y-%m-%d#%H-%M-%S')

    # create logger to file and console
    logging_path = os.path.join(exp_path, 'logging')
    set_logger(logging_folder=logging_path, verbose=args.verbose)
    logging_paths = {
        'checkpoint': os.path.join(exp_path, 'checkpoints'),
        'test_rlt_csv': os.path.join(exp_path, 'test_results', 'results_%s.csv' % time_str),
        'test_rlt_folder': os.path.join(exp_path, 'test_results', 'results_%s' % time_str),
        'tensorboard': os.path.join(exp_path, 'tensorboard_summary'),
    }
    console_logger = logging.getLogger('PL_testing')
    console_logger.info('Total number of tasks: %d' % len(config))
    tb_logger = TensorBoardLogger(logging_paths['tensorboard'], name='', version='', default_hp_metric=False)

    rlt = []
    for task_conf in config:
        console_logger.info('Begin to run task %s' % task_conf['dataset']['kwargs']['task_id'])
        task_conf['num_workers'] = 0  # sequentially load
        # test_metrics = ['dice', 'dice_subject', 'dice_sample', 'hd_subject', 'hd_sample']
        test_metrics = ['dice', 'dice_sample']
        if 'test' in task_conf and 'metric' in task_conf['test']:
            for m in task_conf['test']['metric']:
                if m not in test_metrics:
                    test_metrics.append(m)

        test_datasets = []
        for test_dataset_conf in task_conf['test_datasets']:
            test_datasets.append(get_dataset(test_dataset_conf['name'], train=False, **test_dataset_conf['kwargs']))
        test_loaders = [torch.utils.data.DataLoader(
            test_dataset, batch_size=task_conf['train']['batch_size'],
            num_workers=task_conf['num_workers'], pin_memory=task_conf['train'].get('pin_memory', False))
            for test_dataset in test_datasets]

        # load model params
        model_file = os.path.join(logging_paths['checkpoint'],
                                  'task_%d' % task_conf['dataset']['kwargs']['task_id'], 'last.ckpt')
        if not os.path.exists(model_file):
            console_logger.critical('No last.ckpt file exists for task %d' % task_conf['dataset']['kwargs']['task_id'])

        model = get_model(task_conf['model']['name'], **task_conf['model']['kwargs'])
        cl_strategy = get_strategy(name=task_conf['strategy']['name'], model=model, task_conf=task_conf,
                                   num_task_epochs=0, **task_conf['strategy']['kwargs'])

        cl_strategy = cl_strategy.load_from_checkpoint(model_file, strict=False,
                                                       model=model, config=task_conf, num_task_epochs=0,
                                                       **task_conf['strategy']['kwargs'])

        trainer = pl.Trainer(gpus=None if args.gpus is None else -1,
                             num_nodes=1,
                             accelerator=None if args.gpus is None else 'ddp',
                             benchmark=True,
                             logger=tb_logger,
                             sync_batchnorm=task_conf['train'].get('sync_batchnorm', False),
                             precision=task_conf['train'].get('precision', 32),
                             )

        results = trainer.predict(cl_strategy, dataloaders=test_loaders)
        inputs, preds, target = [], [], []

        for dataloader_rlt in results:
            inputs.append(torch.cat([batch[0] for batch in dataloader_rlt]))
            preds.append(torch.cat([batch[1] for batch in dataloader_rlt]))
            target.append(torch.cat([batch[2] for batch in dataloader_rlt]))

        num_classes = preds[0].shape[1]
        rlt_steps = {'step': task_conf['dataset']['kwargs']['task_id']}
        for test_idx in range(len(test_loaders)):
            inp, p, t = inputs[test_idx], preds[test_idx], target[test_idx]
            assert len(inp) == len(p) == len(t)

            subject_idx = [test_datasets[test_idx].get_subject_item_id(i)[0] for i in range(len(p))]
            slice_idx = [test_datasets[test_idx].get_subject_item_id(i)[1] for i in range(len(p))]
            inp_sub, p_sub, t_sub = cluster(inp, p, t, cluster_keys=subject_idx, internal_orders=slice_idx)
            inp_sub = [item.unsqueeze(0) for item in inp_sub]
            p_sub = [torch.transpose(item, 0, 1).unsqueeze(0) for item in p_sub]
            t_sub = [item.unsqueeze(0) for item in t_sub]

            if args.save_results:
                results_folder = os.path.join(logging_paths['test_rlt_folder'],
                                              'task_%d' % task_conf['dataset']['kwargs']['task_id'],
                                              'data_%d' % (test_idx + 1))
                if not os.path.exists(results_folder):
                    os.makedirs(results_folder)
                for sub_idx, (inp_sub_, p_sub_, t_sub_) in enumerate(zip(inp_sub, p_sub, t_sub)):
                    inp_sub_ = ((inp_sub_.mean(dim=2).squeeze(0)) * 255).type(torch.int16).detach().cpu().numpy()
                    p_sub_ = p_sub_.argmax(dim=1).type(torch.int16).squeeze(0).detach().cpu().numpy()
                    t_sub_ = t_sub_.type(torch.int16).squeeze(0).detach().cpu().numpy()
                    inp_sub_ = sitk.GetImageFromArray(inp_sub_)
                    p_sub_ = sitk.GetImageFromArray(p_sub_)
                    t_sub_ = sitk.GetImageFromArray(t_sub_)
                    sitk.WriteImage(inp_sub_, os.path.join(results_folder, 'subject_%03d_input.nii.gz' % sub_idx),
                                    useCompression=True)
                    sitk.WriteImage(p_sub_, os.path.join(results_folder, 'subject_%03d_pred.nii.gz' % sub_idx),
                                    useCompression=True)
                    sitk.WriteImage(t_sub_, os.path.join(results_folder, 'subject_%03d_target.nii.gz' % sub_idx),
                                    useCompression=True)

            metric_rlt = {}
            for metric in test_metrics:
                if metric == 'dice':
                    metric_cls = Dice(num_classes=num_classes).to(p.device)
                    metric_cls.update(p, t)
                    metric_rlt['dice'] = format_tensor(metric_cls.compute())
                elif metric == 'dice_subject':
                    metric_cls = DiceSampleWise(class_id=None, bg=False).to(p.device)
                    for p_sub_, t_sub_ in zip(p_sub, t_sub):
                        metric_cls.update(p_sub_, t_sub_)
                    metric_rlt['dice_subject'] = format_tensor(metric_cls.compute())
                    for class_id in range(1, num_classes):
                        metric_cls = DiceSampleWise(class_id=class_id).to(p.device)
                        for p_sub_, t_sub_ in zip(p_sub, t_sub):
                            metric_cls.update(p_sub_, t_sub_)
                        metric_rlt[
                            'dice_subject_class_%d' % class_id] = format_tensor(metric_cls.compute())
                elif metric == 'dice_sample':
                    metric_cls = DiceSampleWise(class_id=None, bg=False).to(p.device)
                    metric_cls.update(p, t)
                    metric_rlt['dice_sample'] = format_tensor(metric_cls.compute())
                    for class_id in range(1, num_classes):
                        metric_cls = DiceSampleWise(class_id=class_id).to(p.device)
                        metric_cls.update(p, t)
                        metric_rlt[
                            'dice_sample_class_%d' % class_id] = format_tensor(metric_cls.compute())
                elif metric == 'hd_subject':
                    for class_id in range(1, num_classes):
                        metric_cls = HausdorffDistance(class_id, store_all=True, inf_score=100).to(p.device)
                        for p_sub_, t_sub_ in zip(p_sub, t_sub):
                            metric_cls.update(p_sub_, t_sub_)
                        metric_rlt[
                            'hd_subject_class_%d' % class_id] = format_tensor(metric_cls.compute())
                elif metric == 'hd_sample':
                    for class_id in range(1, num_classes):
                        metric_cls = HausdorffDistance(class_id, store_all=True, inf_score=100).to(p.device)
                        metric_cls.update(p, t)
                        metric_rlt['hd_sample_class_%d' % class_id] = format_tensor(metric_cls.compute())
                else:
                    raise AttributeError
            for key, value in metric_rlt.items():
                rlt_steps['test_%d/' % (test_idx + 1) + key] = value
        rlt.append(rlt_steps)

    rlt_file = logging_paths['test_rlt_csv']
    if not os.path.exists(os.path.dirname(rlt_file)):
        os.makedirs(os.path.dirname(rlt_file))
    with open(rlt_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rlt[0].keys()))
        writer.writeheader()
        writer.writerows(rlt)


def cluster(*args, cluster_keys=None, internal_orders=None):
    assert cluster_keys is not None and internal_orders is not None
    assert isinstance(cluster_keys, (list, tuple)) and isinstance(internal_orders, (list, tuple))
    for x in args:
        assert len(x) == len(cluster_keys) == len(internal_orders)

    ys = [dict() for _ in range(len(args))]
    for i, k in enumerate(cluster_keys):
        for j in range(len(args)):
            if k in ys[j]:
                ys[j][k].append({'x': args[j][i], 'idx': internal_orders[i]})
            else:
                ys[j][k] = [{'x': args[j][i], 'idx': internal_orders[i]}]
    for i in range(len(ys)):
        for k in ys[i].keys():
            ys[i][k].sort(key=lambda item: item['idx'])
            ys[i][k] = torch.stack([item['x'] for item in ys[i][k]])
        ys[i] = list(ys[i].values())
    return *ys,


def format_tensor(tensor):
    t = tensor.detach().cpu().numpy().tolist()
    if isinstance(t, float):
        t = [t]
    s = ' '.join(['%1.5g' % item for item in t])
    return s


if __name__ == '__main__':
    predict_test()
