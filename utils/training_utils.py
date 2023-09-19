import bisect

import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.utils.data


def create_optimizer(params, opt_config):
    optimizer = getattr(torch.optim, opt_config['name'])(params, **opt_config['kwargs'])

    if opt_config.get('lr_scheduler', None) is None:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        sch_conf = opt_config['lr_scheduler']
        if sch_conf['name'] == 'MultiTask':
            lr_scheduler = MultiTaskLR(optimizer, sch_conf.get(['num_epochs_per_task'], None),
                                       **sch_conf['kwargs'])
        else:
            lr_scheduler_cls = getattr(torch.optim.lr_scheduler, sch_conf['name'])
            lr_scheduler = lr_scheduler_cls(optimizer, **sch_conf['kwargs'])

    return optimizer, lr_scheduler


def re_normalize_value_range(tensor, input_range=None, output_range=(0., 1.)):
    if input_range is None:
        input_range = (tensor.min(), tensor.max())
    dtype = tensor.dtype
    out = tensor.type(torch.float32)
    out = (tensor - input_range[0]) / (input_range[1] - input_range[0])
    out = out * (output_range[1] - output_range[0]) + output_range[0]
    out = out.type(dtype)
    return out


class MultiTaskLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, num_epochs_per_task, num_decay_steps=1, gamma_in_task=0.9, gamma_among_tasks=2.0,
                 last_epoch=-1, verbose=False):
        self.optimizer = optimizer

        if num_epochs_per_task is None:
            raise AttributeError('num_epochs_per_task should be set in MultiTaskLR')

        self.num_epochs_per_task = num_epochs_per_task
        self.num_decay_steps = num_decay_steps
        self.gamma_in_task = gamma_in_task
        self.gamma_among_tasks = gamma_among_tasks
        self.epoch_count = 0

        lr_lambdas = [self.lr_lambda_func] * len(optimizer.param_groups)
        super().__init__(optimizer, lr_lambdas, last_epoch, verbose)

    def lr_lambda_func(self, epoch):
        if epoch // self.num_epochs_per_task == 0:
            self.epoch_count = 0
            return self.gamma_among_tasks
        else:
            self.epoch_count += 1
            if self.epoch_count // self.num_decay_steps == 0:
                return self.gamma_in_task
            else:
                return 1.0


class ConcatDataset(torch.utils.data.Dataset):
    datasets = []
    cumulative_sizes = []
    self_attrs = ['self_attrs', 'datasets', 'cumulative_sizes', 'shuffle', 'idx_mapping', 'cumsum']

    def __init__(self, datasets, shuffle=False):
        super().__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(d, torch.utils.data.IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.shuffle = shuffle
        self.idx_mapping = list(range(self.cumulative_sizes[-1]))
        if shuffle:
            rng = np.random.default_rng(seed=1234)
            rng.shuffle(self.idx_mapping)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        idx = self.idx_mapping[idx]

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def __getattr__(self, item):
        return getattr(self.datasets[0], item)

    def clear_cache(self):
        for d in self.datasets:
            d.clear_cache()


def safe_optimizer_step(optimizer, optimizer_closure, skip_opt_step_at_loss_zero=True):
    assert optimizer_closure is not None
    loss = optimizer_closure()  # train_step() -> opt.zero_grad() -> loss.backward()
    # can also be done by configure_gradient_clipping in higher version of pytorch_lightning
    if skip_opt_step_at_loss_zero and not torch.is_nonzero(loss):
        return
    for pg in optimizer.param_groups:
        for p in pg['params']:
            if p.grad is not None:
                p.grad.data.nan_to_num_(nan=0, posinf=1e5, neginf=-1e5)
    optimizer.step()


def get_data_aug_pipe(rgb=True):
    aug_pipe = iaa.Sequential([
        # color, bright
        iaa.Sometimes(0.5, iaa.SomeOf(n=(1, 2), children=[
            iaa.Invert(0.5),
            iaa.MultiplyHueAndSaturation((0.5, 1.5)) if rgb else iaa.Identity(),
            iaa.MultiplyBrightness((0.7, 1.3)) if rgb else iaa.Identity(),
            iaa.GammaContrast((0.7, 1.7)),
        ])),
        # blur, noise
        iaa.Sometimes(0.3, iaa.SomeOf(n=(1, 3), children=[
            iaa.MotionBlur(k=(3, 4)),
            iaa.GaussianBlur(sigma=(0.0, 0.5)),
            iaa.Sharpen(),
            iaa.AdditiveGaussianNoise(scale=(0.01 * 255, 0.05 * 255)),
            iaa.JpegCompression(compression=(75, 99)),
        ])),
        # geometry
        iaa.Sometimes(0.5, iaa.SomeOf(n=(1, 3), children=[
            iaa.HorizontalFlip(0.5),
            iaa.VerticalFlip(0.5),
            iaa.Rotate((-90, 90)),
            iaa.Sequential([iaa.ScaleX((0.7, 1.3)), iaa.ScaleY((0.7, 1.3))]),
            iaa.Sequential([iaa.TranslateX((-0.2, 0.2)), iaa.TranslateY((-0.2, 0.2))]),
            iaa.Sequential([iaa.ShearX((-15, 15)), iaa.ShearY((-15, 15))])
        ]))
    ])
    return aug_pipe
