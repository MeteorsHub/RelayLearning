import argparse
import logging
import os
import re
import time

import imgaug as ia
import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import get_dataset
from models import get_model
from strategies import get_strategy
from utils.exp_logger import ExpLogger
from utils.project_utils import load_config, process_config, save_config, set_logger
from utils.training_utils import ConcatDataset

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='fundus/relay', help='Which config is loaded from configs')
parser.add_argument('-g', '--gpus', type=str, default=None, help='Gpu ids to use, like "1,2,3". If None, use cpu')
parser.add_argument('-n', '--note', type=str, default='default',
                    help='Note to identify this experiment, like "first_version"... Should not contain space')
parser.add_argument('-v', '--verbose', action='store_true', help='If set, log debug level, else info level')
args = parser.parse_args()

if args.gpus is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
exp_path = os.path.join('./exps', args.config, args.note)


def main():
    # torch.autograd.set_detect_anomaly(True)
    config_file = os.path.join('configs', args.config + '.yaml')
    ori_config = load_config(config_file)
    config = process_config(ori_config)

    # backup config
    save_config(os.path.join(exp_path, 'original_config_back.yaml'), ori_config)
    save_config(os.path.join(exp_path, 'processed_config.yaml'), config)
    # create logger to file and console
    logging_path = os.path.join(exp_path, 'logging')
    set_logger(logging_folder=logging_path, verbose=args.verbose)
    logging_paths = {
        'tensorboard': os.path.join(exp_path, 'tensorboard_summary'),
        'checkpoint': os.path.join(exp_path, 'checkpoints'),
        'exp_info_csv': os.path.join(exp_path, 'exp_results')
    }
    console_logger = logging.getLogger('PL_training')
    tb_logger = TensorBoardLogger(logging_paths['tensorboard'], name='', version='', default_hp_metric=False)
    exp_info_logger = ExpLogger(logging_paths['exp_info_csv'], name='', version='')

    console_logger.info('Total number of tasks: %d' % len(config))

    for task_conf in config:
        console_logger.info('Begin to run task %s' % task_conf['dataset']['kwargs']['task_id'])
        if task_conf['random_seed'] is not None:
            seed = task_conf['random_seed'] + task_conf['dataset']['kwargs']['task_id']
            pl.seed_everything(seed, workers=True)
            ia.seed(seed)

        # joint training and validation set
        if task_conf['train'].get('joint_training', False):
            train_datasets = []
            val_datasets = []
            for task_conf_ in config:
                train_datasets.append(get_dataset(
                    task_conf_['dataset']['name'], train=True, **task_conf_['dataset']['kwargs']))
                val_datasets.append(get_dataset(
                    task_conf_['dataset']['name'], train=False, **task_conf_['dataset']['kwargs']))
            train_dataset = ConcatDataset(train_datasets, shuffle=True)
            val_dataset = ConcatDataset(val_datasets, shuffle=False)
        else:
            train_dataset = get_dataset(
                task_conf['dataset']['name'], train=True, **task_conf['dataset']['kwargs'])
            val_dataset = get_dataset(
                task_conf['dataset']['name'], train=False, **task_conf['dataset']['kwargs'])

        test_datasets = []
        for test_dataset_conf in task_conf['test_datasets']:
            test_datasets.append(get_dataset(
                test_dataset_conf['name'], train=False, **test_dataset_conf['kwargs']))

        # prepare loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=task_conf['train']['batch_size'],
                                                   shuffle=True, num_workers=task_conf['num_workers'],
                                                   pin_memory=task_conf['train'].get('pin_memory', False),
                                                   prefetch_factor=task_conf['train'].get('prefetch', 2))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=task_conf['train']['batch_size'],
                                                 shuffle=False, num_workers=task_conf['num_workers'],
                                                 pin_memory=task_conf['train'].get('pin_memory', False),
                                                 prefetch_factor=task_conf['train'].get('prefetch', 2))
        test_loaders = [torch.utils.data.DataLoader(
            test_dataset, batch_size=task_conf['train']['batch_size'], shuffle=False,
            num_workers=task_conf['num_workers'], pin_memory=task_conf['train'].get('pin_memory', False))
            for test_dataset in test_datasets]
        num_task_epochs = task_conf['train'].get('num_epochs', None)
        num_task_mimgs = task_conf['train'].get('num_mimgs', None)
        assert (num_task_epochs is None) ^ (num_task_mimgs is None)
        if num_task_mimgs is not None:
            num_task_epochs = ((num_task_mimgs * 1000 * 1000) // len(train_loader.dataset)) + 1

        additional_model_kwargs = {}
        if 'solver_steps_ratio' not in task_conf['model']['kwargs']:
            if task_conf['train'].get('solver_num_mimgs', None) is not None:
                assert num_task_mimgs is not None
                additional_model_kwargs['solver_steps_ratio'] = task_conf['train']['solver_num_mimgs'] / num_task_mimgs
            elif task_conf['train'].get('solver_num_epochs', None) is not None:
                assert num_task_epochs is not None
                additional_model_kwargs['solver_steps_ratio'] = \
                    task_conf['train']['solver_num_epochs'] / num_task_epochs
        model = get_model(task_conf['model']['name'], **dict(**task_conf['model']['kwargs'], **additional_model_kwargs))

        if model.task != train_dataset.task:
            console_logger.critical('Model task %s is different from dataset task %s'
                                    % (model.task, train_dataset.task))
            exit(-1)
        cl_strategy = get_strategy(name=task_conf['strategy']['name'], model=model, task_conf=task_conf,
                                   num_task_epochs=num_task_epochs, **task_conf['strategy']['kwargs'])

        load_file = None
        resume_file = None
        if task_conf['checkpoint'].get('resume', None) is not None:
            resume_file = os.path.join(logging_paths['checkpoint'], task_conf['checkpoint']['resume'])
        elif task_conf['checkpoint'].get('load', None) is not None:
            load_file = os.path.join(logging_paths['checkpoint'], task_conf['checkpoint']['load'])

        save_dir = os.path.join(logging_paths['checkpoint'], task_conf['checkpoint']['save']['dir'])
        if os.path.exists(save_dir):
            if resume_file is None:
                if 'last.ckpt' in os.listdir(save_dir):
                    console_logger.warning('Task %d seems to be finished because last.ckpt exists. Try next task'
                                           % task_conf['dataset']['kwargs']['task_id'])
                    continue
                ckpt_files = [f for f in os.listdir(save_dir) if f.endswith('.ckpt') and 'last' not in f
                              and 'epoch=' in f]
                if len(ckpt_files) > 0:
                    ckpt_files = sorted(ckpt_files, key=lambda s: int(re.match(r'.*epoch=(\d+).*', s).group(1)))
                    resume_file = os.path.join(save_dir, ckpt_files[-1])
                    console_logger.warning('Model ckpt %s detected in save dir and will be used for resuming'
                                           % resume_file)

        current_task_max_epochs = num_task_epochs
        if resume_file is not None:
            state = torch.load(resume_file, map_location=torch.device('cpu'))
            all_tasks_max_epochs_ = state['all_tasks_max_epochs']
            current_task_max_epochs_ = state['current_task_max_epochs']
            if current_task_max_epochs < current_task_max_epochs_:
                console_logger.warning('num_epoch in config is smaller than that from resume file')
                exit(0)
            else:
                all_tasks_max_epochs = all_tasks_max_epochs_ - current_task_max_epochs_ + current_task_max_epochs
            del state
        elif load_file is not None:
            state = torch.load(load_file, map_location=torch.device('cpu'))
            all_tasks_max_epochs_ = state['all_tasks_max_epochs']
            all_tasks_actual_epochs_ = state['epoch']
            last_task_max_epochs_ = state['current_task_max_epochs']
            if all_tasks_actual_epochs_ != all_tasks_max_epochs_:
                console_logger.warning('Training of the last task is not finished, '
                                       'which is supposed to train %d epochs but only trained %d epochs.'
                                       'We will skip and continue to train current task.'
                                       % (last_task_max_epochs_, all_tasks_actual_epochs_
                                          - (all_tasks_max_epochs_ - last_task_max_epochs_)))
                all_tasks_max_epochs_ = all_tasks_actual_epochs_
            all_tasks_max_epochs = all_tasks_max_epochs_ + current_task_max_epochs
            del state
        else:
            all_tasks_max_epochs = current_task_max_epochs

        checkpoint_saving_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(logging_paths['checkpoint'], task_conf['checkpoint']['save']['dir']),
            every_n_val_epochs=task_conf['train'].get('check_val_every_n_epoch', 1),
        )
        trainer = pl.Trainer(gpus=None if args.gpus is None else -1,
                             num_nodes=1,
                             accelerator=None if args.gpus is None else 'ddp',
                             benchmark=True,

                             max_epochs=all_tasks_max_epochs,
                             sync_batchnorm=task_conf['train'].get('sync_batchnorm', False),
                             gradient_clip_val=task_conf['train']['gradient_clip'],
                             accumulate_grad_batches=task_conf['train'].get('accumulate_grad_batches', 1),
                             precision=task_conf['train'].get('precision', 32),

                             callbacks=[checkpoint_saving_callback],
                             # plugins=pl.plugins.DDPPlugin(find_unused_parameters=False),
                             resume_from_checkpoint=resume_file,
                             check_val_every_n_epoch=task_conf['train'].get('check_val_every_n_epoch', 1),
                             logger=[tb_logger, exp_info_logger],
                             log_every_n_steps=task_conf['train']['log_every_n_steps'],
                             reload_dataloaders_every_epoch=task_conf['train'].get('reload_dataloaders_every_epoch',
                                                                                   False),
                             fast_dev_run=task_conf.get('fast_dev_run', False),
                             num_sanity_val_steps=0,
                             move_metrics_to_cpu=True,
                             )
        if load_file is not None:
            state = torch.load(load_file, map_location=torch.device('cpu'))
            trainer.checkpoint_connector.restore_model_state(cl_strategy, state)
            trainer.checkpoint_connector.restore_training_state(state, load_optimizer_states=False)

        console_logger.info('Begin to train and val task %s' % task_conf['dataset']['kwargs']['task_id'])
        console_logger.info('Current task num_epochs: %d. Total num_epochs: %d'
                            % (current_task_max_epochs, all_tasks_max_epochs))
        trainer.fit(cl_strategy, train_dataloader=train_loader, val_dataloaders=val_loader)
        console_logger.info('Begin to test %d viewed tasks after training of task %s'
                            % (len(task_conf['test_datasets']), task_conf['dataset']['kwargs']['task_id']))

        trainer.test(cl_strategy, test_dataloaders=test_loaders, ckpt_path=None)
        trainer.save_checkpoint(
            os.path.join(logging_paths['checkpoint'], task_conf['checkpoint']['save']['dir'], 'last.ckpt'))
        time.sleep(5)


if __name__ == '__main__':
    main()
