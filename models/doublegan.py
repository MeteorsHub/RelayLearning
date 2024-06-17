import copy

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils.data
import torchmetrics
import torchvision

from losses.distillation import distillation_loss
from models import DeepLearningRelayModel
from models.stylegan2_ada_spade.augment import AugmentPipe, augpipe_specs
from models.stylegan2_ada_spade.misc import assert_shape
from models.stylegan2_ada_spade.networks import Discriminator, Generator
from models.stylegan2_ada_spade.ops import conv2d_gradfix
from utils.project_utils import random_split_samples
from utils.training_utils import create_optimizer, get_data_aug_pipe, re_normalize_value_range
from utils.visualization import gray2rgb, pallate_img_tensor, text_on_imgs


class DoubleGAN(DeepLearningRelayModel):
    # buffers
    pl_mean_img: torch.Tensor
    pl_mean_label: torch.Tensor
    ada_D_signs_real_img: torch.Tensor
    ada_D_signs_real_label: torch.Tensor
    merge_weight: torch.Tensor
    cur_stage: torch.Tensor
    cs_previous_task = dict()  # {'task_name': {'task_name': str, 'id': int, 'frequency': int}
    cs_current_task = dict()  # {'task_name': {'task_name': str, 'id': int, 'frequency': int}

    def __init__(self, optimizer_conf: dict, task='segmentation',
                 img_resolution=512, img_channels=1, label_num_classes=3,
                 # DoubleGAN
                 z_img_dim=512, z_label_dim=512, c_img_dim=0, c_label_dim=0, w_img_dim=512, w_label_dim=256,
                 G_img_kwargs={}, G_label_kwargs={}, D_img_kwargs={}, D_label_kwargs={},
                 # solver
                 solver_name='u-net', solver_kwargs={},
                 # train
                 num_stages=2, weighted_merge=False, label_noise=0.05, style_mixing_prob=0.2,
                 train_gan=True, train_solver=True, solver_steps_ratio=None,
                 # replay
                 replay_w_truncation=1, replay_keep_batch_size=False, replay_data_ratio='half',
                 replay_c_select_mode='uniform', task_merge_replay=True, replay_post_aug=False,
                 # loss
                 G_reg_interval=4, D_reg_interval=16, pl_weight=2, r1_gamma=10, solver_distil_weight=0, distil_tau=2.0,
                 # ada
                 aug_conf_img='bgc', ada_target_img=0.6, aug_p_range_img=None,
                 aug_conf_label='bgc', ada_target_label=0.6, aug_p_range_label=None,
                 ada_interval=4, ada_kimg=500
                 ):
        super().__init__()
        self.task = task
        self.optimizer_conf = optimizer_conf
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_num_classes = label_num_classes

        self.z_img_dim, self.z_label_dim = z_img_dim, z_label_dim
        self.c_img_dim, self.c_label_dim = c_img_dim, c_label_dim
        assert (c_img_dim > 0 and c_label_dim > 0) or (c_img_dim == 0 and c_label_dim == 0)
        self.w_img_dim, self.w_label_dim = w_img_dim, w_label_dim
        self.G_img_kwargs, self.G_label_kwargs = G_img_kwargs, G_label_kwargs
        self.D_img_kwargs, self.D_label_kwargs = D_img_kwargs, D_label_kwargs
        self.solver_name, self.solver_kwargs = solver_name, solver_kwargs

        self.num_stages = num_stages
        assert num_stages in [1, 2, 3]
        cur_stage = torch.ones([], dtype=torch.int32)
        self.register_buffer('cur_stage', cur_stage)
        self.weighted_merge = weighted_merge
        merge_weight = torch.tensor(0., dtype=torch.float32)
        self.register_buffer('merge_weight', merge_weight)
        self.label_noise = label_noise
        self.style_mixing_prob = style_mixing_prob
        self.train_gan = train_gan
        self.train_solver = train_solver
        self.solver_steps_ratio = solver_steps_ratio if self.train_solver else None

        self.replay_w_truncation = replay_w_truncation
        self.replay_keep_batch_size = replay_keep_batch_size
        self.replay_data_ratio = replay_data_ratio
        assert replay_data_ratio in ['half', 'even']
        self.replay_c_select_mode = replay_c_select_mode
        self.task_merge_replay = task_merge_replay
        self.replay_post_aug_pipe = get_data_aug_pipe(rgb=True) if replay_post_aug else None

        self.G_reg_interval, self.D_reg_interval = G_reg_interval, D_reg_interval
        self.pl_weight, self.r1_gamma = pl_weight, r1_gamma
        self.pl_batch_shrink, self.pl_decay = 2, 0.01
        pl_mean_img, pl_mean_label = torch.zeros([]), torch.zeros([])
        self.register_buffer('pl_mean_img', pl_mean_img)
        self.register_buffer('pl_mean_label', pl_mean_label)
        self.solver_distil_weight, self.distil_tau = solver_distil_weight, distil_tau

        self.aug_conf_img, self.aug_conf_label = aug_conf_img, aug_conf_label
        self.ada_target_img, self.ada_target_label = ada_target_img, ada_target_label
        self.aug_p_range_img = aug_p_range_img if aug_p_range_img is not None else [0., 1.]
        self.aug_p_range_label = aug_p_range_label if aug_p_range_label is not None else [0., 1.]
        self.ada_interval = ada_interval
        self.ada_kimg = ada_kimg

        self.augment_pipe_img = AugmentPipe(**augpipe_specs[aug_conf_img]).requires_grad_(False)
        self.augment_pipe_img.p.fill_(0.)
        self.register_buffer('ada_D_signs_real_img', torch.tensor(0., dtype=torch.float32))
        self.augment_pipe_label = AugmentPipe(**augpipe_specs[aug_conf_label]).requires_grad_(False)
        self.augment_pipe_label.p.fill_(0.)
        self.register_buffer('ada_D_signs_real_label', torch.tensor(0., dtype=torch.float32))

        self.G_img = Generator(z_img_dim, c_img_dim, w_img_dim, img_resolution, img_channels, use_spade=True,
                               spade_label_nc=1, **G_img_kwargs)
        self.G_label = Generator(z_label_dim, c_label_dim, z_label_dim, img_resolution, 1, **G_label_kwargs)
        self.D_img = Discriminator(c_img_dim, img_resolution, img_channels + 1, **D_img_kwargs)
        self.D_label = Discriminator(c_label_dim, img_resolution, 1, **D_label_kwargs)

        from models import get_model
        self.solver = get_model(solver_name, **solver_kwargs)

        self.G_img_old = copy.deepcopy(self.G_img).eval().requires_grad_(False)
        self.G_label_old = copy.deepcopy(self.G_label).eval().requires_grad_(False)
        self.D_img_old = copy.deepcopy(self.D_img).eval().requires_grad_(False)
        self.D_label_old = copy.deepcopy(self.D_label).eval().requires_grad_(False)
        self.solver_old = copy.deepcopy(self.solver).eval().requires_grad_(False)

        self.zero_loss = torch.nn.Parameter(torch.tensor([0.]))

        self.fid_img = torchmetrics.FID(feature=2048)
        self.fid_label = torchmetrics.FID(feature=2048)

    def configure_optimizers(self):
        g_params = list(filter(lambda p: p.requires_grad, self.G_img.parameters())) + \
                   list(filter(lambda p: p.requires_grad, self.G_label.parameters()))
        d_params = list(filter(lambda p: p.requires_grad, self.D_img.parameters())) + \
                   list(filter(lambda p: p.requires_grad, self.D_label.parameters()))
        m_params = filter(lambda p: p.requires_grad, self.solver.parameters())

        g_opt, g_lr_sch = create_optimizer(g_params, self.optimizer_conf['G'])
        d_opt, d_lr_sch = create_optimizer(d_params, self.optimizer_conf['D'])
        m_opt, m_lr_sch = create_optimizer(m_params, self.optimizer_conf['M'])

        g_spec = {'optimizer': g_opt, 'lr_scheduler': {'scheduler': g_lr_sch, 'name': 'g_lr'}}
        d_spec = {'optimizer': d_opt, 'lr_scheduler': {'scheduler': d_lr_sch, 'name': 'd_lr'}}
        m_spec = {'optimizer': m_opt, 'lr_scheduler': {'scheduler': m_lr_sch, 'name': 'm_lr'}}
        return g_spec, d_spec, m_spec

    def training_step(self, batch, batch_idx, optimizer_idx=None, **kwargs):
        if not self.train_gan and optimizer_idx in [0, 1]:
            return 0. * self.zero_loss
        if not self.train_solver and optimizer_idx == 2:
            return 0. * self.zero_loss

        real_img, real_label = batch
        real_batch_size = len(real_img)

        _, real_cs_img, _, real_cs_label = self.generate_zs_cs(len(real_img), 1, c_bank='current', device=self.device)
        task_merge_replay, task_split_replay, old_g_replay = False, False, True
        if self.num_stages == 1 and self.base_strategy.task_id > 1:
            task_split_replay = True
        elif self.num_stages == 2 and self.base_strategy.task_id > 1 and self.cur_stage == 2:
            task_split_replay = True
        elif self.num_stages == 3:
            if self.cur_stage == 2:
                task_split_replay = True
            if self.cur_stage == 3:
                task_split_replay = True
                task_merge_replay = False if not self.task_merge_replay else True
                old_g_replay = False
        if task_split_replay or task_merge_replay:
            with torch.no_grad():
                real_img, real_label, real_cs_img, real_cs_label, real_batch_size = \
                    self.data_merge_with_replay(
                        real_img, real_label, real_cs_img, real_cs_label, task_split=task_split_replay,
                        task_merge=task_merge_replay, old=old_g_replay)

        real_label_1c = self.label_id_to_1c(real_label)
        if self.label_noise > 0:
            real_label_1c = self.noisy_label(real_label_1c)

        pl = (self.base_strategy.global_step + 1) % self.G_reg_interval == 0
        r1 = (self.base_strategy.global_step + 1) % self.D_reg_interval == 0

        if optimizer_idx in [0, 1] and self.cur_stage in [1, 2]:
            if self.style_mixing_prob > 0 and np.random.rand() < self.style_mixing_prob:
                num_styles = 2
            else:
                num_styles = 1
            c_bank = 'current' if self.cur_stage == 1 else 'previous+current'
            fake_zs_img, fake_cs_img, fake_zs_label, fake_cs_label = self.generate_zs_cs(
                len(real_img), num_styles, c_bank=c_bank, single_cid=True, device=self.device)
            gen_spade_in = real_label_1c
            img_of_gen_spade_in = real_img

        if optimizer_idx == 0:  # G
            if self.cur_stage == 3:
                loss = 0. * self.zero_loss
            else:
                loss_G_img = self.loss_G(fake_zs_img, fake_cs_img, g='img', spade_in=gen_spade_in,
                                         img_of_spade_in=img_of_gen_spade_in, img_of_spade_in_cs=real_cs_img,
                                         pl=pl, phase='train', batch_idx=batch_idx)
                loss_G_label = self.loss_G(fake_zs_label, fake_cs_label, g='label', pl=pl,
                                           phase='train', batch_idx=batch_idx)
                loss = loss_G_img + loss_G_label
        elif optimizer_idx == 1:  # D
            if self.cur_stage == 3:
                loss = 0. * self.zero_loss
            else:
                loss_D_img = self.loss_D(fake_zs_img, fake_cs_img, real_img, real_cs_img, d='img',
                                         gen_spade_in=gen_spade_in, real_spade_in=real_label_1c, r1=r1,
                                         phase='train', batch_idx=batch_idx)
                loss_D_label = self.loss_D(fake_zs_label, fake_cs_label, real_label_1c, real_cs_label, d='label',
                                           r1=r1, phase='train', batch_idx=batch_idx)
                loss = loss_D_img + loss_D_label
        elif optimizer_idx == 2:  # M
            if self.base_strategy.binary_positive_label is not None:
                real_label = (real_label == self.base_strategy.binary_positive_label).type(torch.int64)
            if self.replay_post_aug_pipe is not None:
                if real_batch_size < len(real_img):
                    aug_img, aug_label = self.replay_post_aug(real_img[real_batch_size:], real_label[real_batch_size:])
                    in_img = torch.cat([real_img[:real_batch_size], aug_img], 0)
                    in_label = torch.cat([real_label[:real_batch_size], aug_label], 0)
                else:
                    in_img, in_label = real_img, real_label
                ori_img, ori_label = real_img, real_label
            else:
                in_img, in_label, ori_img, ori_label = real_img, real_label, None, None

            loss = self.loss_M(in_img, in_label, real_cs_img, real_cs_label, phase='train',
                               ori_img=ori_img, ori_label=ori_label, batch_idx=batch_idx)
        else:
            raise RuntimeError('optimizer_idx %d is not recognized' % optimizer_idx)
        return loss

    def start_new_task(self):
        with torch.no_grad():
            # update old models
            models = [self.G_img, self.G_label, self.D_img, self.D_label, self.solver]
            old_models = [self.G_img_old, self.G_label_old, self.D_img_old, self.D_label_old, self.solver_old]
            for model, old_model in zip(models, old_models):
                for p, p_old in zip(model.parameters(), old_model.parameters()):
                    p_old.copy_(p)
                for b, b_old in zip(model.buffers(), old_model.buffers()):
                    b_old.copy_(b)
            # stage reset
            self.cur_stage.fill_(1)
            # half the aug_p
            self.augment_pipe_img.p.div_(2.)
            self.augment_pipe_label.p.div_(2.)
            # update cs_previous_task
            self.cs_previous_task = self.cs_dict_merge(self.cs_previous_task, self.cs_current_task)
            self.cs_current_task = dict()

    def on_train_epoch_start(self) -> None:
        previous_tasks_num_epochs = self.base_strategy.trainer.max_epochs - self.base_strategy.num_task_epochs
        cur_epochs_of_cur_task = self.base_strategy.trainer.current_epoch - previous_tasks_num_epochs
        if self.num_stages in [2, 3]:
            # update cur_stage
            assert self.base_strategy.num_task_epochs > self.num_stages
            if self.num_stages == 2:
                if cur_epochs_of_cur_task == self.base_strategy.num_task_epochs // 2:
                    self.model_merge()
                    if self.base_strategy.task_id > 1:  # no stage 2 in task_1
                        self.cur_stage.fill_(2)
            elif self.num_stages == 3:
                if cur_epochs_of_cur_task == self.base_strategy.num_task_epochs // 3:
                    self.model_merge()
                    if self.base_strategy.task_id > 1:  # no stage 2 in task_1
                        self.cur_stage.fill_(2)
                if cur_epochs_of_cur_task == (2 * self.base_strategy.num_task_epochs) // 3:
                    self.cur_stage.fill_(3)
        # train_solver
        if self.solver_steps_ratio is not None:
            assert 0.0 <= self.solver_steps_ratio <= 1.0
            solver_num_epochs = int(round(self.base_strategy.num_task_epochs * self.solver_steps_ratio))
            solver_epoch_ids = np.linspace(1, self.base_strategy.num_task_epochs, solver_num_epochs, dtype=np.int) \
                .tolist()
            if cur_epochs_of_cur_task in solver_epoch_ids:
                self.train_solver = True
            else:
                self.train_solver = False

    def on_train_batch_start(self, batch, batch_idx: int, dataloader_idx: int) -> None:
        # update cs_current_task
        if self.cur_stage == 1:
            self.cs_current_task_add(self.base_strategy.task_name, num_samples=len(batch[0]))

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int) -> None:
        with torch.no_grad():
            batch_size = len(batch[0])
            world_size = self.base_strategy.trainer.accelerator.training_type_plugin.world_size

            if self.cur_stage != 3:
                # buffer sync
                if self.base_strategy.trainer.accelerator.training_type_plugin.distributed_backend == 'ddp':
                    if world_size > 1:
                        buffers_need_reduce_mean = [self.pl_mean_img, self.pl_mean_label,
                                                    self.ada_D_signs_real_img, self.ada_D_signs_real_label]
                        for buffer in buffers_need_reduce_mean:
                            torch.distributed.all_reduce(buffer, op=torch.distributed.ReduceOp.SUM)
                            buffer.div_(world_size)
                # update augment_pipe
                if (batch_idx + 1) % self.ada_interval == 0:
                    adjust_img = torch.sign(self.ada_D_signs_real_img / self.ada_interval - self.ada_target_img) * \
                                 (batch_size * world_size * self.ada_interval) / (self.ada_kimg * 1000)
                    self.augment_pipe_img.p.copy_(torch.clip((self.augment_pipe_img.p + adjust_img),
                                                             self.aug_p_range_img[0], self.aug_p_range_img[1]))
                    self.ada_D_signs_real_img.zero_()
                    adjust_label = torch.sign(self.ada_D_signs_real_label / self.ada_interval - self.ada_target_label) \
                                   * (batch_size * world_size * self.ada_interval) / (self.ada_kimg * 1000)
                    self.augment_pipe_label.p.copy_(torch.clip((self.augment_pipe_label.p + adjust_label),
                                                               self.aug_p_range_label[0], self.aug_p_range_label[1]))
                    self.ada_D_signs_real_label.zero_()

    def validation_step(self, batch, batch_idx, **kwargs):
        return self.validation_test_step('val', batch, batch_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=None, **kwargs):
        return self.validation_test_step('test', batch, batch_idx, dataloader_idx)

    def validation_test_step(self, phase, batch, batch_idx, dataloader_idx=None):
        with torch.no_grad():
            assert phase in ['val', 'test']
            real_img, real_label = batch
            _, real_cs_img, _, real_cs_label = self.generate_zs_cs(len(real_img), 1,
                                                                   c_bank='current', device=self.device)
            loss = 0. * self.zero_loss
            if self.train_gan or phase == 'test':
                real_label_1c = self.label_id_to_1c(real_label)
                if self.label_noise > 0:
                    real_label_1c = self.noisy_label(real_label_1c)

                if self.style_mixing_prob > 0 and np.random.rand() < self.style_mixing_prob:
                    num_styles = 2
                else:
                    num_styles = 1
                fake_zs_img, fake_cs_img, fake_zs_label, fake_cs_label = self.generate_zs_cs(
                    len(real_img), num_styles, c_bank='current', single_cid=True, device=self.device)

                gen_label, _ = self.forward_G(fake_zs_label, fake_cs_label, g='label')
                loss_G_img = self.loss_G(fake_zs_img, fake_cs_label, g='img',
                                         spade_in=gen_label, img_of_spade_in=torch.zeros_like(real_img),
                                         img_of_spade_in_cs=real_cs_img,
                                         pl=False, phase=phase, batch_idx=batch_idx)
                loss_D_img = self.loss_D(fake_zs_img, fake_cs_img, real_img, real_cs_img, d='img',
                                         gen_spade_in=gen_label,
                                         real_spade_in=real_label_1c, r1=False, phase=phase, batch_idx=batch_idx)
                loss += loss_D_img + loss_G_img

                loss_G_label = self.loss_G(fake_zs_label, fake_cs_label, g='label', pl=False, phase=phase,
                                           batch_idx=batch_idx)
                loss_D_label = self.loss_D(fake_zs_label, fake_cs_label, real_label_1c, real_cs_label, d='label',
                                           r1=False, phase=phase, batch_idx=batch_idx)
                loss += loss_D_label + loss_G_label
            if self.train_solver or phase == 'test':
                if self.base_strategy.binary_positive_label is not None:
                    real_label = (real_label == self.base_strategy.binary_positive_label).type(torch.int64)
                loss_M = self.loss_M(real_img, real_label, real_cs_img, real_cs_label,
                                     phase=phase, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
                loss += loss_M
        return loss

    def model_merge(self):
        # model merging
        if self.weighted_merge:
            cur_weight = self.base_strategy.trainer.train_dataloader.dataset.datasets.num_samples
        else:
            cur_weight = 1.
        if self.base_strategy.task_id != 1:
            with torch.no_grad():
                old_models = [self.G_img_old, self.G_label_old, self.D_img_old, self.D_label_old, self.solver_old]
                cur_models = [self.G_img, self.G_label, self.D_img, self.D_label, self.solver]
                for old_model, cur_model in zip(old_models, cur_models):
                    for p_old, p_cur in zip(old_model.parameters(), cur_model.parameters()):
                        weight = self.merge_weight / (self.merge_weight + cur_weight)
                        p_cur.lerp_(p_old, weight)
        # update merge weight
        self.merge_weight.add_(cur_weight)

    def merge_cs(self, cs):
        if self.c_img_dim == 0:
            return cs.sum(dim=1)
        if cs.ndim == 3:
            c = cs.sum(dim=1)
            max_c, _ = c.max(dim=1, keepdim=True)
            c = c / max_c
        elif cs.ndim == 2:
            c = cs
        else:
            raise RuntimeError('cs ndim is %d' % cs.ndim)
        return c

    def generate_zs_cs(self, num_samples, num_styles, c_bank='previous', c_select_mode='uniform',
                       single_cid=False, device=None):
        # final zs: [num_samples, num_styles, z_dim]
        # final cs: [num_samples, num_styles, c_dim]
        assert c_bank in ['previous', 'current', 'previous+current'] or isinstance(c_bank, list)
        assert c_select_mode in ['uniform', 'distribution']
        zs_img = torch.randn([num_samples, num_styles, self.z_img_dim], device=device)
        zs_label = torch.randn([num_samples, num_styles, self.z_label_dim], device=device)
        if self.c_img_dim == 0:
            cs_img = torch.zeros([num_samples, num_styles, self.c_img_dim], device=device)
            cs_label = torch.zeros([num_samples, num_styles, self.c_label_dim], device=device)
        else:
            if c_bank == 'previous':
                c_bank = list(self.cs_previous_task.values())
            elif c_bank == 'current':
                c_bank = list(self.cs_current_task.values())
            else:  # previous+current
                c_bank = list(self.cs_dict_merge(self.cs_previous_task, self.cs_current_task).values())
            if c_select_mode == 'uniform':
                select_weight = None
            else:  # from num_samples distribution
                select_weight = [v['frequency'] for v in c_bank]
                select_weight = [v / sum(select_weight) for v in select_weight]

            if single_cid:
                cs_id_img = np.random.choice([v['id'] for v in c_bank], [num_samples, 1], replace=True, p=select_weight)
                cs_id_img = np.tile(cs_id_img, (1, num_styles))
                cs_id_label = cs_id_img
            else:
                cs_id_img = np.random.choice([v['id'] for v in c_bank], [num_samples, num_styles], replace=True,
                                             p=select_weight)
                cs_id_label = np.random.choice([v['id'] for v in c_bank], [num_samples, num_styles], replace=True,
                                               p=select_weight)

            def _cids2cs(_c_id, _c_dim):
                _cs_id = torch.tensor(_c_id, dtype=torch.long, device=device)
                _cs_flat = F.one_hot(_cs_id.view([-1]), num_classes=_c_dim)
                _cs = _cs_flat.view([num_samples, num_styles, _c_dim])
                return _cs

            cs_img, cs_label = _cids2cs(cs_id_img, self.c_img_dim), _cids2cs(cs_id_label, self.c_label_dim)
        return zs_img, cs_img, zs_label, cs_label

    def cs_dict_merge(self, dict1: dict, dict2: dict) -> dict:
        new_dict = copy.deepcopy(dict1)
        for k, v in dict2.items():
            if k in new_dict:
                new_dict[k]['frequency'] += v['frequency']
            else:
                new_dict[k] = copy.deepcopy(v)
        return new_dict

    def cs_current_task_add(self, task_name: str, num_samples: int):
        if task_name in self.cs_current_task:
            self.cs_current_task[task_name]['frequency'] += num_samples
        elif task_name in self.cs_previous_task:
            self.cs_current_task[task_name] = {
                'task_name': task_name,
                'id': self.cs_previous_task[task_name]['id'],
                'frequency': num_samples
            }
        else:
            new_id = 0
            for v in list(self.cs_previous_task.values()) + list(self.cs_current_task.values()):
                if v['id'] >= new_id:
                    new_id = v['id'] + 1
            self.cs_current_task[task_name] = {
                'task_name': task_name,
                'id': new_id,
                'frequency': num_samples
            }

    def get_task_name_from_cs(self, cs) -> list:
        # cs: [num_samples, num_styles, c_dim] note that cs are not merged, means they are split
        if self.c_img_dim == 0:
            return ['' for _ in range(len(cs))]
        _, c_ids = cs.max(dim=2)
        c_bank = list(self.cs_dict_merge(self.cs_previous_task, self.cs_current_task).values())
        c_id_name_map = dict()
        for item in c_bank:
            c_id_name_map[item['id']] = item['task_name']
        task_names = []
        for c_id in c_ids:
            task_name = [c_id_name_map[cid.item()] for cid in c_id]
            task_name = '\n'.join(task_name)
            task_names.append(task_name)
        return task_names

    def forward_G(self, zs, cs, style_mixing_num_ws='random', g='img', spade_in=None, truncation_psi=1, old=False):
        assert g in ['img', 'label']
        assert style_mixing_num_ws in ['random', 'even'] or isinstance(style_mixing_num_ws, list)
        if g == 'img':
            assert spade_in is not None
            G = self.G_img_old if old else self.G_img
        else:
            assert spade_in is None
            G = self.G_label_old if old else self.G_label

        if zs.ndim == 2:
            zs = zs.unsqueeze(1)
        if cs.ndim == 2:
            cs = zs.unsqueeze(1)
        assert zs.shape[1] == cs.shape[1]
        num_styles = zs.shape[1]
        if style_mixing_num_ws == 'even':
            style_mixing_num_ws = [G.num_ws // num_styles for _ in range(num_styles - 1)]
            style_mixing_num_ws.append(G.num_ws - len(style_mixing_num_ws))
        elif style_mixing_num_ws == 'random':
            style_mixing_num_ws = random_split_samples(G.num_ws, num_styles, True)
        assert len(style_mixing_num_ws) == num_styles
        assert sum(style_mixing_num_ws) == G.num_ws
        assert_shape(zs, [None, num_styles, G.z_dim])
        assert_shape(cs, [None, num_styles, G.c_dim])

        ws = []
        for style_i in range(num_styles):
            ws.append(G.mapping(zs[:, style_i], cs[:, style_i],
                                truncation_psi=truncation_psi)[:, :style_mixing_num_ws[style_i]])
        ws = torch.cat(ws, 1)

        gen = G.synthesis(ws, spade_in)
        return gen, ws

    def forward_D(self, inputs, cs, d='img', spade_in=None, return_aug=False, aug=True, old=False):
        assert d in ['img', 'label']
        if old:
            assert not aug
        if d == 'img':
            assert spade_in is not None
            inputs = torch.cat([inputs, spade_in], 1)
            if aug:
                inputs = self.augment_pipe_img(inputs)
            D = self.D_img_old if old else self.D_img
        else:
            assert spade_in is None
            if aug:
                inputs = self.augment_pipe_label(inputs)
            D = self.D_label_old if old else self.D_label
        c = self.merge_cs(cs)
        logits = D(inputs, c)
        if return_aug:
            return logits, inputs
        return logits

    def forward_M(self, img, old=False):
        solver = self.solver_old if old else self.solver
        pred_logits = solver(img)
        pred_logits = pred_logits.to(torch.float32)
        return pred_logits

    def forward(self, img):
        return self.forward_M(img)

    def replay(self, num_samples, task_split=True, task_merge=False, old=True):
        assert task_split ^ task_merge
        with torch.no_grad():
            num_styles = 1 if task_split else 2
            c_bank = 'previous' if old else 'previous+current'
            fake_zs_img, fake_cs_img, fake_zs_label, fake_cs_label = self.generate_zs_cs(
                num_samples, num_styles=num_styles, c_bank=c_bank,
                c_select_mode=self.replay_c_select_mode, single_cid=task_split, device=self.device)
            gen_label, _ = self.forward_G(fake_zs_label, fake_cs_label, style_mixing_num_ws='random', g='label',
                                          truncation_psi=self.replay_w_truncation, old=old)
            gen_img, _ = self.forward_G(fake_zs_img, fake_cs_img, style_mixing_num_ws='random',
                                        g='img', spade_in=gen_label, truncation_psi=self.replay_w_truncation, old=old)

            # clip the replay img and label to -1~1
            gen_img = gen_img.clip(-1, 1)
            gen_label = gen_label.clip(-1, 1)
        return gen_img, gen_label, fake_cs_img, fake_cs_label

    def replay_post_aug(self, img, label_id):
        assert self.replay_post_aug_pipe is not None
        device = img.device
        with torch.no_grad():
            img = re_normalize_value_range(img, input_range=(-1., 1.), output_range=(0., 255.))
            img = img.detach().cpu().numpy().astype(np.uint8).transpose((0, 2, 3, 1))
            label_id = label_id.detach().cpu().numpy().astype(np.int32)
            aug_img, aug_label = self.replay_post_aug_pipe(images=img, segmentation_maps=np.expand_dims(label_id, -1))
            aug_label = aug_label.squeeze(-1)
            aug_img = aug_img.transpose(0, 3, 1, 2).astype(np.float32)
            aug_img = torch.tensor(aug_img, dtype=torch.float32, device=device)
            aug_label = torch.tensor(aug_label, dtype=torch.long, device=device)
            aug_img = re_normalize_value_range(aug_img, input_range=(0., 255.), output_range=(-1., 1.))
        return aug_img, aug_label

    def data_merge_with_replay(self, real_img, real_label_id, real_img_cs, real_label_cs, task_split=True,
                               task_merge=False, old=True):
        assert task_split or task_merge
        assert len(real_label_id) == len(real_img)

        with torch.no_grad():
            batch_size = len(real_img)
            num_parts_replay = int(task_split) + int(task_merge)
            if self.replay_keep_batch_size:
                if self.replay_data_ratio == 'half':
                    replay_batch_size_per_part = batch_size // (num_parts_replay + 1)
                else:  # self.replay_data_ratio == 'even'
                    replay_batch_size_per_part = \
                        int((batch_size / num_parts_replay) * (1 - 1 / (1 + len(self.cs_previous_task))))
                real_batch_size = batch_size - replay_batch_size_per_part * num_parts_replay
            else:
                if self.replay_data_ratio == 'half':
                    replay_batch_size_per_part = batch_size
                else:
                    replay_batch_size_per_part = batch_size * len(self.cs_previous_task)
                real_batch_size = batch_size

            if replay_batch_size_per_part <= 0 or real_batch_size <= 0:
                return real_img, real_label_id, real_img_cs, real_label_cs
            if real_batch_size < batch_size:
                real_idx = torch.multinomial(torch.ones([batch_size], device=self.device),
                                             num_samples=real_batch_size, replacement=False)
                real_img, real_label_id = real_img[real_idx], real_label_id[real_idx]
                real_img_cs, real_label_cs = real_img_cs[real_idx], real_label_cs[real_idx]

            all_imgs, all_label_ids = [real_img], [real_label_id]
            all_cs_img, all_cs_label = [real_img_cs], [real_label_cs]
            if task_split:
                replay_img, replay_label, replay_cs_img, replay_cs_label = self.replay(
                    replay_batch_size_per_part, task_split=True, task_merge=False, old=old)
                replay_label_id = self.label_1c_to_id(replay_label)
                all_imgs.append(replay_img)
                all_label_ids.append(replay_label_id)
                all_cs_img.append(replay_cs_img)
                all_cs_label.append(replay_cs_label)
            if task_merge:
                replay_img, replay_label, replay_cs_img, replay_cs_label = \
                    self.replay(replay_batch_size_per_part, task_split=False, task_merge=True, old=old)
                replay_label_id = self.label_1c_to_id(replay_label)
                all_imgs.append(replay_img)
                all_label_ids.append(replay_label_id)
                all_cs_img.append(replay_cs_img)
                all_cs_label.append(replay_cs_label)
                # num_styles unify only on m_training of stage3
                max_num_styles = replay_cs_img.shape[1]
                if max_num_styles > 1:
                    for i in range(len(all_cs_img) - 1):
                        all_cs_img[i] = torch.tile(all_cs_img[i], (1, max_num_styles, 1))
                        all_cs_label[i] = torch.tile(all_cs_label[i], (1, max_num_styles, 1))

            batch_img, batch_label_id = torch.cat(all_imgs), torch.cat(all_label_ids)
            batch_cs_img, batch_cs_label = torch.cat(all_cs_img), torch.cat(all_cs_label)
            return batch_img, batch_label_id, batch_cs_img, batch_cs_label, real_batch_size

    def loss_G(self, zs, cs, g='img', spade_in=None, img_of_spade_in=None, img_of_spade_in_cs=None,
               pl=False, phase=None, batch_idx=None):
        assert phase in ['train', 'val', 'test']
        assert g in ['img', 'label']
        if g == 'img':
            assert spade_in is not None and img_of_spade_in is not None and img_of_spade_in_cs is not None
            spade_in = spade_in.detach().requires_grad_(False)
            img_of_spade_in = img_of_spade_in.detach().requires_grad_(False)
        else:
            assert spade_in is None and img_of_spade_in is None

        gen, _ = self.forward_G(zs, cs, g=g, spade_in=spade_in)
        gen_logits = self.forward_D(gen, cs, d=g, spade_in=spade_in)

        loss_G = F.softplus(-gen_logits).mean()
        loss = loss_G

        log_kwargs = {
            'G_%s_scores_fake' % g: gen_logits.mean(),
            'G_%s_signs_fake' % g: gen_logits.sign().mean(),
            'loss_G_%s' % g: loss_G
        }

        # path length regularization
        if pl and self.pl_weight != 0:
            assert phase == 'train', 'pl reg should be use only in training'
            pl_batch_size = max(1, len(zs) // self.pl_batch_shrink)
            spade_in_pl = None if spade_in is None else spade_in[:pl_batch_size]
            gen_pl, gen_ws_pl = self.forward_G(zs[:pl_batch_size], cs[:pl_batch_size], g=g, spade_in=spade_in_pl)
            pl_noise = torch.randn_like(gen_pl) / np.sqrt(gen_pl.shape[2] * gen_pl.shape[3])
            with conv2d_gradfix.no_weight_gradients():
                pl_grads = torch.autograd.grad(outputs=[(gen_pl * pl_noise).sum()], inputs=[gen_ws_pl],
                                               create_graph=True, only_inputs=True)[0]
            pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
            if g == 'img':
                pl_mean = self.pl_mean_img.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean_img.copy_(pl_mean.detach())
            else:
                pl_mean = self.pl_mean_label.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean_label.copy_(pl_mean.detach())
            pl_penalty = (pl_lengths - pl_mean).square()
            loss_pl = pl_penalty * self.pl_weight
            loss_pl = loss_pl.mean().mul(self.G_reg_interval)
            loss += loss_pl

            log_kwargs['G_%s_pl_penalty' % g] = pl_penalty.mean()
            log_kwargs['loss_G_%s_reg' % g] = loss_pl

        # log images
        if self.base_strategy.should_log(phase, batch_idx):
            with torch.no_grad():
                if g == 'img':
                    spade_in_id = self.label_1c_to_id(spade_in)
                    spade_in_vis = pallate_img_tensor(spade_in_id, input_range=(0, 255), output_range=(-1, 1))
                    if self.c_img_dim != 0:
                        cs_img = text_on_imgs(torch.zeros_like(spade_in_vis), self.get_task_name_from_cs(cs))
                        cs_img_of_spade_in = text_on_imgs(torch.zeros_like(spade_in_vis),
                                                          self.get_task_name_from_cs(img_of_spade_in_cs))
                        cated_img = torch.cat([cs_img, gray2rgb(gen),
                                               gray2rgb(img_of_spade_in), spade_in_vis, cs_img_of_spade_in], 0)
                    else:
                        cated_img = torch.cat([gray2rgb(gen),
                                               gray2rgb(img_of_spade_in), spade_in_vis], 0)
                    grid_name = 'GenImg_RealImg_RealLabel'
                else:
                    gen_label = self.label_1c_to_id(gen)
                    label_vis = pallate_img_tensor(gen_label, input_range=(0, 255), output_range=(-1, 1))
                    if self.c_img_dim != 0:
                        cs_img = text_on_imgs(torch.zeros_like(label_vis), self.get_task_name_from_cs(cs))
                        cated_img = torch.cat([cs_img, label_vis], 0)
                    else:
                        cated_img = torch.cat([label_vis], 0)
                    grid_name = 'GenLabel'
                grid = torchvision.utils.make_grid(cated_img, nrow=len(gen),
                                                   normalize=True, value_range=(-1, 1), pad_value=1)
                log_additional_images = {grid_name: grid}
                self.base_strategy.log_assets(phase, additional_images=log_additional_images)
        if self.base_strategy.should_log(phase, batch_idx):
            self.base_strategy.log_assets(phase, additional_log_kwargs=log_kwargs)
        return loss

    def loss_D(self, zs, cs, real_inputs, real_cs, d='img', gen_spade_in=None, real_spade_in=None, r1=False, phase=None,
               batch_idx=None):
        assert phase in ['train', 'val', 'test']
        assert d in ['img', 'label']
        if d == 'img':
            assert gen_spade_in is not None and real_spade_in is not None
            gen_spade_in = gen_spade_in.detach().requires_grad_(False)
            real_spade_in = real_spade_in.detach().requires_grad_(False)
        else:
            assert gen_spade_in is None and real_spade_in is None

        r1 = r1 and self.r1_gamma != 0
        if r1:
            assert not torch.is_nonzero(self.zero_loss)
            real_inputs = real_inputs + self.zero_loss  # to get real_inputs requires_grad and avoid memory leak

        gen, gen_ws = self.forward_G(zs, cs, g=d, spade_in=gen_spade_in)
        gen_logits, gen_aug_inputs = self.forward_D(gen, cs, d=d, spade_in=gen_spade_in, return_aug=True)
        loss_Dgen = F.softplus(gen_logits).mean()

        real_logits, real_aug_inputs = self.forward_D(real_inputs, real_cs, d=d, spade_in=real_spade_in,
                                                      return_aug=True)
        loss_Dreal = F.softplus(-real_logits).mean()
        loss = loss_Dgen + loss_Dreal

        if phase == 'train' and d == 'img':
            self.ada_D_signs_real_img.add_(real_logits.sign().mean())
        if phase == 'train' and d == 'label':
            self.ada_D_signs_real_label.add_(real_logits.sign().mean())

        log_kwargs = {
            'D_scores_fake_%s' % d: gen_logits.mean(),
            'D_signs_fake_%s' % d: gen_logits.sign().mean(),
            'D_scores_real_%s' % d: real_logits.mean(),
            'D_signs_real_%s' % d: real_logits.sign().mean(),
            'loss_D_%s' % d: loss,
        }
        if d == 'img':
            log_kwargs['aug_p_img'] = self.augment_pipe_img.p
        if d == 'label':
            log_kwargs['aug_p_label'] = self.augment_pipe_label.p

        # r1 regularization
        if r1 and self.r1_gamma != 0:
            with conv2d_gradfix.no_weight_gradients():
                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_inputs], create_graph=True,
                                               only_inputs=True)[0]
            r1_penalty = r1_grads.square().sum([1, 2, 3])
            loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
            loss_Dr1 = loss_Dr1.mean().mul(self.D_reg_interval)
            loss += loss_Dr1

            log_kwargs['D_r1_penalty_%s' % d] = r1_penalty.mean()
            log_kwargs['loss_D_reg_%s' % d] = loss_Dr1

        # log FID
        if phase in ['val', 'test'] and self.cur_stage != 3:
            if d == 'img':
                self.fid_img.update(gray2rgb(gen, dtype=torch.uint8), real=False)
                self.fid_img.update(gray2rgb(real_inputs, dtype=torch.uint8), real=True)
                self.base_strategy.log('%s_metric/fid_img' % phase,
                                       self.fid_img, on_step=False, on_epoch=True, sync_dist=True)
            else:
                self.fid_label.update(gray2rgb(gen, dtype=torch.uint8), real=False)
                self.fid_label.update(gray2rgb(real_inputs, dtype=torch.uint8), real=True)
                self.base_strategy.log('%s_metric/fid_label' % phase,
                                       self.fid_label, on_step=False, on_epoch=True, sync_dist=True)

        # log images
        if self.base_strategy.should_log(phase, batch_idx, log_img=True):
            with torch.no_grad():
                img_grids = []
                img_grids_names = []
                if d == 'img':
                    gen_aug_img = gen_aug_inputs[:, :self.img_channels]
                    gen_aug_label_of_img = gen_aug_inputs[:, self.img_channels:]
                    gen_aug_label_of_img = self.label_1c_to_id(gen_aug_label_of_img)
                    gen_aug_label_of_img_vis = \
                        pallate_img_tensor(gen_aug_label_of_img, input_range=(0, 255), output_range=(-1, 1))
                    real_aug_inputs = real_aug_inputs[:, :self.img_channels]
                    if self.c_img_dim != 0:
                        cs_img = text_on_imgs(torch.zeros_like(gen_aug_label_of_img_vis),
                                              self.get_task_name_from_cs(cs))
                        img_grids += [cs_img, gray2rgb(gen_aug_img), gen_aug_label_of_img_vis,
                                      gray2rgb(real_aug_inputs)]
                    else:
                        img_grids += [gray2rgb(gen_aug_img), gen_aug_label_of_img_vis, gray2rgb(real_aug_inputs)]
                    img_grids_names.append('GenImgAug_GenLabelOfImgAug_RealImgAug')
                if d == 'label':
                    gen_aug_label = self.label_1c_to_id(gen_aug_inputs)
                    gen_aug_label_vis = pallate_img_tensor(gen_aug_label, input_range=(0, 255), output_range=(-1, 1))
                    real_aug_label = self.label_1c_to_id(real_aug_inputs)
                    real_aug_label_vis = pallate_img_tensor(real_aug_label, input_range=(0, 255), output_range=(-1, 1))
                    if self.c_img_dim != 0:
                        cs_img = text_on_imgs(torch.zeros_like(gen_aug_label_vis), self.get_task_name_from_cs(cs))
                        img_grids += [cs_img, gen_aug_label_vis, real_aug_label_vis]
                    else:
                        img_grids += [gen_aug_label_vis, real_aug_label_vis]
                    img_grids_names.append('GenLabelAug_RealLabelAug')
                if len(img_grids) > 0:
                    img_grid = torchvision.utils.make_grid(torch.cat(img_grids, 0), nrow=len(gen_aug_inputs),
                                                           normalize=True, value_range=(-1, 1), pad_value=1)
                    log_additional_imgs = {'_'.join(img_grids_names): img_grid}
                else:
                    log_additional_imgs = None
                self.base_strategy.log_assets(phase, additional_images=log_additional_imgs)
        if self.base_strategy.should_log(phase, batch_idx):
            self.base_strategy.log_assets(phase, additional_log_kwargs=log_kwargs)
        return loss

    def loss_M(self, real_img, real_label_id, real_cs_img, real_cs_label, phase: str, ori_img=None, ori_label=None,
               batch_idx=None, dataloader_idx=None):
        assert phase in ['train', 'val', 'test']

        pred_logits = self.forward_M(real_img).type(torch.float32)
        loss = self.base_strategy.criterion(pred_logits, real_label_id)

        if self.solver_distil_weight > 0 and self.base_strategy.task_id > 1:
            with torch.no_grad():
                teacher_logits = self.forward_M(real_img).type(torch.float32)
            loss_dist = self.solver_distil_weight * distillation_loss(teacher_logits, pred_logits, tau=self.distil_tau)
            loss += loss_dist

        if batch_idx is not None:
            img_metas = []
            if self.base_strategy.should_log(phase, batch_idx, log_img=True):
                if self.c_img_dim != 0:
                    real_c_str_img = self.get_task_name_from_cs(real_cs_img)
                    real_c_str_label = self.get_task_name_from_cs(real_cs_label)
                    real_c_str = ['i)%s\nl)%s' % (img_str, label_str)
                                  for img_str, label_str in zip(real_c_str_img, real_c_str_label)]
                    img_metas.append((text_on_imgs(torch.zeros_like(gray2rgb(real_img)), real_c_str).type(
                        torch.float32)) / 255.)
                if ori_img is not None:
                    img_metas.append(ori_img)
                if ori_label is not None:
                    img_metas.append(pallate_img_tensor(ori_label, input_range=(0, 255), output_range=(-1, 1)))
            self.base_strategy.shared_step_log(phase, batch_idx, real_img, pred_logits, real_label_id,
                                               img_metas=img_metas, dataloader_idx=dataloader_idx)
            if self.base_strategy.should_log(phase, batch_idx):
                log_kwargs = {}
                if self.solver_distil_weight > 0 and self.base_strategy.task_id > 1:
                    log_kwargs['loss_M_distillation'] = loss_dist
                log_kwargs['cur_stage'] = float(self.cur_stage)
                prog_bar_kwargs = {'stage': self.cur_stage}
                self.base_strategy.log_assets(phase, prog_bar_kwargs=prog_bar_kwargs, additional_log_kwargs=log_kwargs)

        return loss

    def label_id_to_1c(self, label_id, num_classes=None):
        assert label_id.ndim == 3  # NHW
        if num_classes is None:
            num_classes = self.label_num_classes
        # 0~label_num_classes-1 to -1~1
        assert label_id.max() < num_classes and label_id.min() >= 0
        label_1c = label_id.type(torch.float32).unsqueeze(1) * (2 / (num_classes - 1)) - 1.  # -1~1
        return label_1c

    def label_1c_to_id(self, label_1c, num_classes=None):
        assert label_1c.ndim == 4  # NCHW, C=1
        if num_classes is None:
            num_classes = self.label_num_classes
        # -1~1 to 0~label_num_classes
        label_1c = label_1c.clip(-1, 1)
        label_id = (label_1c.squeeze(1) + 1.) * ((num_classes - 1) / 2)
        label_id = torch.round(label_id).type(torch.int64)
        return label_id

    def noisy_label(self, label):
        with torch.no_grad():
            label = label + self.label_noise * torch.randn_like(label).clip(-3, 3)
        return label

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint['RL_meta'] = {
            'cs_previous_task': self.cs_previous_task,
            'cs_current_task': self.cs_current_task
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.cs_previous_task = checkpoint['RL_meta']['cs_previous_task']
        self.cs_current_task = checkpoint['RL_meta']['cs_current_task']
