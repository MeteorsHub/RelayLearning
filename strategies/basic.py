import logging
from abc import ABC
from typing import List

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sn
import torch
import torchmetrics
import torchvision

from losses import get_loss
from models.basic import DeepLearningModel
from utils.metrics import Dice, DiceSampleWise, HausdorffDistance, confidence_interval_boostrap, confidence_interval_t
from utils.training_utils import create_optimizer, re_normalize_value_range, safe_optimizer_step
from utils.visualization import fig_to_numpy, gray2rgb, pallate_img_tensor_uint8


class MultiSiteLearningStrategy(pl.LightningModule, ABC):
    """
    Multi-site learning strategy for one task at a time in a sequence.
    """
    name: str = None

    task: None
    num_classes: None
    task_id: int = None
    task_name: str = None

    previous_task_id: torch.Tensor
    before_all_tasks_flag: torch.Tensor
    first_task_flag: torch.Tensor
    task_final_epoch_flag: torch.Tensor

    def __init__(self, model: DeepLearningModel, config: dict, num_task_epochs: int,
                 strict_load_state_dict: bool = None,
                 freeze_module_names: List[str] = None,
                 load_state_dict_from_ckpt: bool = True,
                 load_state_dict_param_prefix: List[str] = None,
                 load_metric_table_from_ckpt: bool = True,
                 binary_positive_label: int = None,
                 rgb_inputs: bool = False,
                 additional_metrics: list[str] = None,
                 **kwargs):
        super().__init__()
        self.model = model
        self.config = config
        self.strict_load_state_dict = strict_load_state_dict
        self.freeze_module_names = freeze_module_names
        self.load_state_dict_from_ckpt = load_state_dict_from_ckpt
        self.load_state_dict_param_prefix = load_state_dict_param_prefix
        self.load_metric_table_from_ckpt = load_metric_table_from_ckpt
        self.binary_positive_label = binary_positive_label
        self.rgb_inputs = rgb_inputs
        self.additional_metrics = additional_metrics

        self.freeze_modules()
        self.model.configure_base_strategy(self)

        self.task = config['dataset']['kwargs']['task']
        self.num_classes = config['dataset']['kwargs']['num_classes'] if binary_positive_label is None else 2
        self.task_id = config['dataset']['kwargs']['task_id']
        self.task_name = config['dataset']['kwargs']['task_name']
        self.num_test_datasets = len(config['test_datasets'])
        self.test_datasets_ids = [conf['kwargs']['task_id'] for conf in config['test_datasets']]
        self.num_task_epochs = num_task_epochs

        self.register_buffer('previous_task_id', torch.tensor([0], dtype=torch.int, device=self.device))
        self.register_buffer('before_all_tasks_flag', torch.tensor([True], dtype=torch.bool, device=self.device))
        self.register_buffer('first_task_flag', torch.tensor([True], dtype=torch.bool, device=self.device))
        self.register_buffer('task_final_epoch_flag', torch.tensor([False], dtype=torch.bool, device=self.device))

        self.criterion = None
        self.configure_criterion()

        self.output_transfer = torch.nn.Softmax(dim=1)

        all_metrics = dict()
        if self.task in ['segmentation', 'classification']:
            all_metrics['acc'] = torchmetrics.Accuracy(num_classes=self.num_classes, average='macro')
        if self.task == 'segmentation':
            all_metrics['dice'] = Dice(num_classes=self.num_classes, threshold=0.5)
        if self.additional_metrics is not None:
            for am in self.additional_metrics:
                if am == 'hd':
                    for class_id in range(1, self.num_classes):
                        if self.num_classes == 2:
                            all_metrics['hd'] = HausdorffDistance()
                        else:
                            all_metrics['hd_class_%d' % class_id] = HausdorffDistance(class_id=class_id)
                elif am == 'hd_all_sample':
                    for class_id in range(1, self.num_classes):
                        all_metrics['hd_all_sample_class_%d' % class_id] = \
                            HausdorffDistance(class_id=class_id, store_all=True)
                elif am == 'dice_class':
                    for class_id in range(self.num_classes):
                        all_metrics['dice_class_%d' % class_id] = Dice(num_classes=self.num_classes, class_id=class_id)
                elif am == 'dice_all_sample':
                    for class_id in range(1, self.num_classes):
                        all_metrics['dice_all_sample_class_%d' % class_id] = DiceSampleWise(class_id=class_id)
                    all_metrics['dice_all_sample'] = DiceSampleWise()
                else:
                    raise AttributeError('metric %s not recognized' % am)

        test_metrics = {}
        for test_task_id in self.test_datasets_ids:
            test_metrics[str(test_task_id)] = torchmetrics.MetricCollection(all_metrics).clone(
                prefix='test_%d_metric/' % test_task_id)
        for k in list(all_metrics.keys()):
            if 'all_sample' in k:
                all_metrics.pop(k)
        # for micro_avg
        test_metrics['micro_avg'] = torchmetrics.MetricCollection(all_metrics).clone(
            prefix='test_metric/', postfix='_micro_avg')
        self.test_metrics = torch.nn.ModuleDict(test_metrics)

        self.train_metrics = torchmetrics.MetricCollection(all_metrics).clone(prefix='train_metric/')
        self.val_metrics = torchmetrics.MetricCollection(all_metrics).clone(prefix='val_metric/')

        # { metric_1: -1*torch.ones([num_trained_tasks+1(random_result), num_test_tasks]), metric_2:...}
        for k in all_metrics.keys():
            self.register_buffer('test_metric_%s_table' % k,
                                 -1 * torch.ones([self.task_id, self.num_test_datasets], device=self.device))

    def freeze_modules(self) -> None:
        if self.freeze_module_names is not None:
            named_modules = dict(self.model.named_modules())
            for module_name in self.freeze_module_names:
                for params in named_modules[module_name].parameters():
                    params.requires_grad = False

    def configure_optimizers(self):
        opt_params = filter(lambda p: p.requires_grad, self.model.parameters())
        opt, lr_sch = create_optimizer(opt_params, self.config['train']['optimizer'])
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': lr_sch, 'name': 'lr'}}

    def configure_criterion(self):
        cri = get_loss(self.config['train']['loss']['name'], device=self.device,
                       **self.config['train']['loss']['kwargs'])
        self.criterion = cri

    def optimizer_step(
            self,
            epoch: int = None,
            batch_idx: int = None,
            optimizer=None,
            optimizer_idx: int = None,
            optimizer_closure=None,
            on_tpu: bool = None,
            using_native_amp: bool = None,
            using_lbfgs: bool = None,
    ) -> None:
        safe_optimizer_step(optimizer, optimizer_closure, skip_opt_step_at_loss_zero=True)

    def forward(self, x, output_transfer=False) -> torch.Tensor:
        x = self.model(x)
        if output_transfer:
            x = self.output_transfer(x)
        return x

    # train on current task
    def training_step(self, batch, batch_idx, optimizer_idx=None) -> torch.Tensor:
        raise NotImplementedError

    # val on current tasks
    def validation_step(self, batch, batch_idx) -> None:
        inputs, labels = batch
        outputs = self.forward(inputs)
        if self.binary_positive_label is not None:
            labels = (labels == self.binary_positive_label).type(torch.int64)
        self.shared_step_log('val', batch_idx, inputs, outputs, labels)

    # test on previous tasks
    def test_step(self, batch, batch_idx, dataloader_idx=None) -> None:
        inputs, labels = batch
        outputs = self.forward(inputs)
        if self.binary_positive_label is not None:
            labels = (labels == self.binary_positive_label).type(torch.int64)
        self.shared_step_log('test', batch_idx, inputs, outputs, labels, dataloader_idx=dataloader_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            if self.binary_positive_label is not None:
                labels = (batch[1] == self.binary_positive_label).type(torch.int64)
            else:
                labels = batch[1]
            return batch[0], self.forward(batch[0], output_transfer=True), labels  # inputs, preds and labels
        else:
            return batch, self.forward(batch, output_transfer=True)  # inputs and preds

    def should_log(self, phase, batch_idx=None, log_img=False):
        assert phase in ['train', 'val', 'test']
        if phase == 'train':
            if not log_img:
                log_flag = self.global_step % self.config['train']['log_every_n_steps'] == 0
            else:
                log_flag = self.global_step % (5 * self.config['train']['log_every_n_steps']) == 0
        else:
            assert batch_idx is not None
            if not log_img:
                log_flag = True
            else:
                log_flag = batch_idx % self.config['train']['log_every_n_steps'] == 0
        return log_flag

    def shared_step_log(self, phase, batch_idx, inputs, outputs, labels,
                        img_metas=None, dataloader_idx=None) -> None:
        # should only be called once in a step. otherwise use log_assets instead
        assert phase in ['train', 'val', 'test']
        if torch.any(torch.isnan(outputs)):
            logging.getLogger('RL').critical('Output is nan, please check the results')
            exit(-1)
        # this transform should be performed by the caller of this function
        # if self.binary_positive_label is not None:
        #     labels = (labels == self.binary_positive_label).type(torch.int64)

        log_flag, log_img_flag = self.should_log(phase, batch_idx), self.should_log(phase, batch_idx, log_img=True)
        if not (log_flag or log_img_flag):
            return

        if self.criterion is not None:
            standard_loss = self.criterion(outputs, labels)
            self.log_assets(phase, standard_loss=standard_loss)
        preds = self.output_transfer(outputs)

        if phase == 'train' and log_flag:
            self.log_assets(phase, log_lr=True, log_task_id=True)
        if self.task == 'segmentation':
            if log_img_flag:
                self.log_assets(phase, batch_idx=batch_idx, dataloader_idx=dataloader_idx,
                                images={'inputs': inputs, 'predlabelmaps': preds, 'labelmaps': labels,
                                        'img_metas': img_metas})
            if log_flag:
                self.log_metrics(phase, preds, labels, dataloader_idx)
        if self.task == 'classification':
            if log_img_flag:
                self.log_assets(phase, batch_idx=batch_idx, dataloader_idx=dataloader_idx,
                                images={'inputs': inputs, 'predlabels': preds, 'labels': labels,
                                        'img_metas': img_metas})
            if log_flag:
                self.log_metrics(phase, preds, labels, dataloader_idx)

    def log_assets(self, phase, batch_idx=None, dataloader_idx=None, standard_loss=None, images=None, log_lr=False,
                   log_task_id=False,
                   prog_bar_kwargs=None, additional_log_kwargs: dict = None, additional_images: dict = None,
                   task_scalar_kwargs: dict = None, task_hist_kwargs: dict = None):
        assert phase in ['train', 'val', 'test']

        if images is not None:
            all_images = []
            names = []
            img_range = self.config['dataset']['kwargs'].get('out_range', None)
            log_range = (0., 1.)
            if 'img_metas' in images and images['img_metas'] is not None:
                if not isinstance(images['img_metas'], (list, tuple)):
                    images['img_metas'] = [images['img_metas']]
                for img_meta in images['img_metas']:
                    img_meta = re_normalize_value_range(img_meta, img_range, log_range)
                    all_images.append(gray2rgb(img_meta))
            if 'inputs' in images:
                inputs = re_normalize_value_range(images['inputs'], img_range, log_range)
                if inputs.shape[1] == 1:  # channel 1 to 3
                    new_shape = [-1] * inputs.ndim
                    new_shape[1] = 3
                    inputs = inputs.expand(*new_shape)
                all_images.append(inputs)
                names.append('inputs')
            if 'gens' in images:
                gens = re_normalize_value_range(images['gens'], img_range, log_range)
                if gens.shape[1] == 1:  # channel 1 to 3
                    new_shape = [-1] * gens.ndim
                    new_shape[1] = 3
                    gens = gens.expand(*new_shape)
                all_images.append(gens)
                names.append('gens')
            if 'predlabelmaps' in images:
                if images['predlabelmaps'].ndim == 4:
                    predlabelmaps = torch.argmax(images['predlabelmaps'], 1, keepdim=True)
                else:
                    predlabelmaps = images['predlabelmaps']
                predlabelmaps = pallate_img_tensor_uint8(predlabelmaps)
                predlabelmaps = re_normalize_value_range(predlabelmaps, (0, 255), log_range)
                all_images.append(predlabelmaps)
                names.append('predlabelmaps')
            if 'labelmaps' in images:
                if images['labelmaps'].ndim == 3:
                    labelmaps = torch.unsqueeze(images['labelmaps'], 1)
                else:
                    labelmaps = images['labelmaps']
                labelmaps = pallate_img_tensor_uint8(labelmaps)
                labelmaps = re_normalize_value_range(labelmaps, (0, 255), log_range)
                all_images.append(labelmaps)
                names.append('labelmaps')
            image_grid = torchvision.utils.make_grid(torch.cat(all_images, 0), nrow=len(all_images[0]),
                                                     normalize=True, value_range=log_range, pad_value=1)
            image_grid = image_grid.detach()
            if image_grid.is_cuda:
                image_grid = image_grid.cpu()
            image_tag = '%s_task%d/%s' % (phase, self.task_id, '_'.join(names))
            if dataloader_idx is not None:
                image_tag = image_tag.replace('/', '/data%d_' % dataloader_idx)
            step = self.global_step if phase != 'test' or batch_idx is None else batch_idx
            self.logger.experiment[0].add_image(image_tag, image_grid,
                                                global_step=step, dataformats='CHW')
        if log_lr:
            last_lrs = self.get_last_lr(with_name=True)
            if not isinstance(last_lrs, list):
                last_lrs = [last_lrs]
            for cl in last_lrs:
                self.log('epoch/%s' % cl['name'], cl['lr'], prog_bar=True, logger=True, on_step=False,
                         on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        if standard_loss is not None:
            if phase == 'val':
                self.log('hp/val_standard_loss', standard_loss, sync_dist=True, add_dataloader_idx=False)
            self.log('%s_stats/standard_loss' % phase, standard_loss, prog_bar=False, logger=True, on_step=None,
                     on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        if log_task_id:
            if phase == 'train':
                self.log('epoch/task_id', float(self.task_id), on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log('task_id', self.task_id, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        if additional_log_kwargs is not None:
            for name, value in additional_log_kwargs.items():
                self.log('%s_stats/%s' % (phase, name), value, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        if prog_bar_kwargs is not None:
            for name, value in prog_bar_kwargs.items():
                self.log(name, value, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        if additional_images is not None:
            for name, value in additional_images.items():
                value = value.detach()
                if value.is_cuda:
                    value = value.cpu()
                image_tag = '%s_task%d/%s' % (phase, self.task_id, name)
                if dataloader_idx is not None:
                    image_tag = image_tag.replace('/', '/data%d_' % dataloader_idx)
                step = self.global_step if phase != 'test' or batch_idx is None else batch_idx
                self.logger.experiment[0].add_image(image_tag, value,
                                                    global_step=step, dataformats='CHW')
        if task_scalar_kwargs is not None:
            for name, value in task_scalar_kwargs.items():
                self.logger.experiment[0].add_scalar('%s_stats/%s' % (phase, name), value, global_step=self.task_id)
        if task_hist_kwargs is not None:
            for name, value in task_hist_kwargs.items():
                self.logger.experiment[0].add_histogram('%s/%s' % (phase, name), value, global_step=self.task_id)

    def log_metrics(self, phase=None, preds=None, targets=None, dataloader_idx=None, on_phase_end=None):
        assert on_phase_end is None or on_phase_end in ['train', 'test']
        # log in steps
        if on_phase_end is None:
            # log preds distribution
            histo_prefix = phase
            if phase in ['train', 'val']:
                histo_prefix += '%s_%d' % (phase, self.task_id)
                self.logger.experiment[0].add_histogram('%s/preds' % histo_prefix, preds, global_step=self.global_step)
                self.logger.experiment[0].add_histogram('%s/targets' % histo_prefix, targets,
                                                        global_step=self.global_step)
            assert phase is not None and preds is not None and targets is not None
            if phase == 'train':
                metrics = [self.train_metrics]
            elif phase == 'val':
                metrics = [self.val_metrics]
            elif phase == 'test':
                assert dataloader_idx is not None or self.num_test_datasets == 1
                if dataloader_idx is None:
                    dataloader_idx = 0
                metrics = [self.test_metrics[str(self.test_datasets_ids[dataloader_idx])],
                           self.test_metrics['micro_avg']]
            else:
                raise AttributeError

            outputs = dict()
            for _metrics in metrics:
                if self.num_classes == 2 and self.task in ['segmentation', 'classification']:
                    for k in _metrics.keys():
                        if 'all_sample' in k:
                            if phase == 'test':
                                _metrics[k.split('/')[-1].replace('_micro_avg', '')].update(preds, targets)
                        else:
                            outputs[k] = _metrics[k.split('/')[-1].replace('_micro_avg', '')](preds, targets)
                elif self.task == 'generation':
                    outputs = dict()
                    for k in _metrics.keys():
                        if 'all_sample' in k:
                            if phase == 'test':
                                _metrics[k.split('/')[-1].replace('_micro_avg', '')].update(preds, targets)
                        else:
                            outputs[k] = _metrics[k.split('/')[-1].replace('_micro_avg', '')](preds, targets)
                else:
                    outputs = _metrics(preds, targets)

            valid_outputs = {}
            for k, v in outputs.items():
                if v is not None:
                    valid_outputs[k] = v
            if phase == 'val':
                if 'val_metric/dice' in valid_outputs:
                    valid_outputs['hp/val_metric_dice'] = valid_outputs['val_metric/dice']
            self.log_dict(valid_outputs, on_step=None, on_epoch=True, add_dataloader_idx=False)
        # log in train/test ends
        else:
            # log the task value of train and val metrics
            if on_phase_end == 'train':
                for phase, metrics in zip(['train', 'val'], [self.train_metrics, self.val_metrics]):
                    for k in metrics:
                        try:
                            v = metrics[k.split('/')[-1]].compute()
                            self.logger.experiment[0].add_scalar('%s_metric/%s_task' % (phase, k), v,
                                                                 global_step=self.task_id)
                        except RuntimeError:
                            logging.getLogger('MultiSiteLearningStrategy'). \
                                warning('No metrics can be computed from phase %s' % phase)
            if on_phase_end == 'test':
                # log the mean test metrics
                test_metric_values_list = [v.compute() for k, v in self.test_metrics.items() if 'micro_avg' not in k]
                test_metric_mean_values_dict = dict()
                for i_test_set, test_metric_values in enumerate(test_metric_values_list):
                    # test_metric_values: [metric_1, metric_2...] on the i-th testset
                    for k, v in test_metric_values.items():
                        if 'all_sample' not in k:
                            metric_table = self.__getattr__('test_metric_%s_table' % k.split('/')[-1])
                            # log task metric
                            self.logger.experiment[0].add_scalar('%s_task' % k, v, global_step=self.task_id)
                            # update test table
                            metric_table[self.task_id - 1, i_test_set].copy_(v)
                            # update mean value computation
                            k = k.split('/')[1:]  # remove test_1_metric/ in metric name
                            k = '/'.join(k)
                            if k in test_metric_mean_values_dict:
                                test_metric_mean_values_dict[k].append(v)
                            else:
                                test_metric_mean_values_dict[k] = [v]
                        else:
                            self.logger.experiment[1].log_info_dict({k: v}, step=self.task_id)
                            mean, (ci_left, ci_right) = confidence_interval_boostrap(v, ci=0.95, num_boostrap=5000)
                            self.logger.experiment[1].log_info_dict(
                                {k.replace('all_sample', 'boostrap_CI'): torch.stack([mean, ci_left, ci_right])},
                                step=self.task_id)
                            mean, (ci_left, ci_right) = confidence_interval_t(v, ci=0.95)
                            self.logger.experiment[1].log_info_dict(
                                {k.replace('all_sample', 't_CI'): torch.stack([mean, ci_left, ci_right])},
                                step=self.task_id)
                # log mean test metric
                for k in test_metric_mean_values_dict:
                    test_metric_mean_values_dict[k] = torch.mean(torch.stack(test_metric_mean_values_dict[k]))
                    self.logger.experiment[0].add_scalar('test_metric/%s_task' % k, test_metric_mean_values_dict[k],
                                                         global_step=self.task_id)
                for k, v in self.test_metrics['micro_avg'].items():
                    self.logger.experiment[0].add_scalar('%s_task' % k,
                                                         v.compute(), global_step=self.task_id)

                # log test tables
                for metric_name, metric_table in self.named_buffers(recurse=False):
                    if 'test_metric' not in metric_name or 'table' not in metric_name:
                        continue
                    metric_name = metric_name.split('_')[-2]

                    fig = plt.Figure()
                    ax = fig.add_subplot(1, 1, 1)
                    sn.heatmap(metric_table.cpu().numpy(), vmin=0, vmax=1, annot=True, fmt='.3f',
                               yticklabels=list(range(1, metric_table.shape[0] + 1)),
                               xticklabels=list(range(1, metric_table.shape[1] + 1)),
                               linewidths=.5, ax=ax)
                    metric_table_img = fig_to_numpy(fig)
                    self.logger.experiment[0].add_image(
                        'test/%s' % metric_name, metric_table_img, global_step=self.task_id, dataformats='CWH')

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            tqdm_dict.pop('v_num')
        return tqdm_dict

    def on_train_start(self):
        if self.previous_task_id == 0 and self.task_id != 1:
            logging.getLogger('ParameterLoading').warning(
                'Train task id should start form 1, which is %d. '
                'If not train separately, please be cautious' % self.task_id)
        if self.previous_task_id != 0:
            if self.task_id == self.previous_task_id:
                logging.getLogger('ParameterLoading'). \
                    warning('Current and previous task_id are the same. If not resuming, please be cautious')
            elif self.task_id != self.previous_task_id + 1:
                logging.getLogger('ParameterLoading'). \
                    warning('Train task id should be %d in order, but is %d. If not train jointly, be cautious'
                            % ((self.previous_task_id + 1).item(), self.task_id))
        if self.before_all_tasks_flag:
            self.logger[0].log_hyperparams(self.hparams, {"hp/val_standard_loss": 0, "hp/val_metric_dice": 0})
        with torch.no_grad():
            if self.current_epoch < self.trainer.max_epochs - 1:
                self.task_final_epoch_flag.zero_()
            else:
                self.task_final_epoch_flag.fill_(True).type(torch.BoolTensor)
            self.previous_task_id.fill_(self.task_id)

    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        with torch.no_grad():
            self.before_all_tasks_flag.zero_()

    def on_train_end(self) -> None:
        self.log_metrics(on_phase_end='train')
        with torch.no_grad():
            self.first_task_flag.zero_()

    def on_test_end(self) -> None:
        self.log_metrics(on_phase_end='test')
        for v in self.test_metrics.values():
            v.reset()

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint['all_tasks_max_epochs'] = self.trainer.max_epochs
        checkpoint['current_task_max_epochs'] = self.num_task_epochs

    def get_current_lr(self, with_name=False):
        if not self.trainer.lr_schedulers:
            return None
        current_lrs = []
        for i, lrs in enumerate(self.trainer.lr_schedulers, 1):
            if len(self.trainer.lr_schedulers) > 1:
                name = lrs.get('name', 'lr_%d' % i)
            else:
                name = lrs.get('name', 'lr')
            lr = lrs['scheduler'].get_lr()[0]  # first param group
            if with_name:
                current_lrs.append({'name': name, 'lr': lr})
            else:
                current_lrs.append(lr)
        if len(current_lrs) == 1:
            return current_lrs[0]
        return current_lrs

    def get_last_lr(self, with_name=False):
        if not self.trainer.lr_schedulers:
            return None
        last_lrs = []
        for i, lrs in enumerate(self.trainer.lr_schedulers):
            if len(self.trainer.lr_schedulers) > 1:
                name = lrs.get('name', 'lr_%d' % i)
            else:
                name = lrs.get('name', 'lr')
            lr = lrs['scheduler'].get_last_lr()[0]  # first param group
            if with_name:
                last_lrs.append({'name': name, 'lr': lr})
            else:
                last_lrs.append(lr)
        if len(last_lrs) == 1:
            return last_lrs[0]
        return last_lrs

    def load_state_dict(self, state_dict, strict: bool = True):
        strict = strict if self.strict_load_state_dict is None else self.strict_load_state_dict
        # load test metrics tables
        table_keys = []
        for key in state_dict.keys():
            if 'test_metric' in key and 'table' in key:
                table_keys.append(key)
        for key in table_keys:
            value = state_dict.pop(key)
            h, w = value.shape
            try:
                hh, ww = self.__getattr__(key).shape
                if self.load_metric_table_from_ckpt:
                    self.__getattr__(key)[:min(h, hh), :min(w, ww)].copy_(value[:min(h, hh), :min(w, ww)])
                state_dict[key] = self.__getattr__(key)
            except AttributeError as e:
                if strict:
                    raise e

        if self.load_state_dict_from_ckpt:
            if self.load_state_dict_param_prefix is not None:
                invalid_keys = []
                for key in state_dict.keys():
                    if not any([key.startswith(p) for p in self.load_state_dict_param_prefix]):
                        invalid_keys.append(key)
                for key in invalid_keys:
                    state_dict.pop(key)
            super().load_state_dict(state_dict, strict)

    def register_buffers(self, buffer_dict: dict, prefix='buffer', persistent=True):
        for buffer_name, buffer in buffer_dict.items():
            name = prefix + '/' + buffer_name.replace('.', '/')
            self.register_buffer(name, buffer, persistent=persistent)

    def get_buffer(self, buffer_name, prefix='buffer'):
        name = prefix + '/' + buffer_name.replace('.', '/')
        return self.__getattr__(name)
