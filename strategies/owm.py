from typing import List

import torch
import torch.nn.functional as F

from models.basic import DeepLearningModel
from strategies.basic import MultiSiteLearningStrategy


class OWM(MultiSiteLearningStrategy):
    name: str = 'owm'

    def __init__(self, model: DeepLearningModel, config: dict, alpha=1.0, a_lambda=0.1, owm_module_names=None,
                 **kwargs):
        super().__init__(model, config, **kwargs)

        self.save_hyperparameters('alpha', 'a_lambda')
        self.a_lambda = a_lambda
        self.alpha = alpha
        self.owm_module_names = owm_module_names

        self.step_alpha = None

        # register hooks
        # only support models without a loop, i.e, each module only forward once
        named_modules = dict(self.model.named_modules())
        named_params = dict(self.model.named_parameters())
        if self.training:
            for module_name, module in named_modules.items():
                if len(list(module.parameters(recurse=False))) != 0:
                    if self.is_module_require_owm(module_name):
                        module.register_forward_hook(self.get_forward_hook(module_name))

        # inputs will be mean value in batch dim
        self.param_names_require_owm = list()
        named_params_inputs = dict()
        named_params_ps = dict()

        # create all ps
        for param_name, param in named_params.items():
            if param is not None and not param.requires_grad:
                continue
            module_name = get_param_direct_module_father_name(param_name)
            if not self.is_module_require_owm(module_name):
                continue

            if module_name in named_modules:
                if isinstance(named_modules[module_name], torch.nn.Conv1d) \
                        or isinstance(named_modules[module_name], torch.nn.Conv2d) \
                        or isinstance(named_modules[module_name], torch.nn.Conv3d) \
                        or isinstance(named_modules[module_name], torch.nn.ConvTranspose2d):
                    if get_param_last_name(param_name) == 'weight':
                        dim = param.shape[1:].numel()
                    elif get_param_last_name(param_name) == 'bias':
                        dim = param.shape.numel()
                    else:
                        raise AttributeError('Unrecognized param %s' % param_name)
                elif isinstance(named_modules[module_name], torch.nn.BatchNorm1d) \
                        or isinstance(named_modules[module_name], torch.nn.BatchNorm2d) \
                        or isinstance(named_modules[module_name], torch.nn.BatchNorm3d):
                    dim = param.shape.numel()
                elif isinstance(named_modules[module_name], torch.nn.Linear):
                    if get_param_last_name(param_name) == 'weight':
                        dim = param.shape[1]
                    elif get_param_last_name(param_name) == 'bias':
                        dim = param.shape.numel()
                    else:
                        raise AttributeError('Unrecognized param' % param_name)
                else:
                    raise AttributeError('model has param %s that is not defined in OWM' % param_name)
                self.param_names_require_owm.append(param_name)
                named_params_ps[param_name] = torch.eye(dim, dtype=torch.float)
                named_params_inputs[param_name] = torch.zeros([1, dim], dtype=torch.float)
            else:
                raise ValueError('Cannot find father module of param %s' % param_name)
        # register all ps in order to save in checkpoint
        self.register_buffers(named_params_ps, prefix='p')
        self.register_buffers(named_params_inputs, prefix='feat_in', persistent=False)

        if isinstance(self.a_lambda, list):
            assert len(self.param_names_require_owm) == len(self.a_lambda), \
                'num_params_require_owm(%d) != num_a_lambda(%d)' \
                % (len(self.param_names_require_owm), len(self.a_lambda))
        else:
            self.a_lambda = [self.a_lambda for _ in range(len(self.param_names_require_owm))]

    def is_module_require_owm(self, module_name):
        if self.owm_module_names is None:
            return True
        need_owm = False
        for owm_module_name in self.owm_module_names:
            if module_name.startswith(owm_module_name):
                need_owm = True
                break
        return need_owm

    def get_forward_hook(self, module_name):
        # to log the inputs
        def hook(module: torch.nn.Module, x, y):
            # if not self.trainer.training:
            if not self.training:
                return
            with torch.no_grad():
                mean_x = torch.mean(x[0].detach(), 0, True)  # only support modules with single input
                mean_y = torch.mean(y.detach(), 0, True)
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
                    if concat_param_names([module_name, 'weight']) in self.param_names_require_owm:
                        # if any([k_size > in_img_size for k_size, in_img_size in
                        #         zip(module.kernel_size, mean_x.shape[2:])]):
                        # logging.getLogger('OWM').warning(
                        #     'Some dim of Conv(%s) input size(%s) smaller than kernel size(%s)'
                        #     % (module_name, mean_x.shape[2:], module.kernel_size))
                        fea_in_ = F.unfold(
                            mean_x,
                            kernel_size=module.kernel_size, padding=module.padding,
                            stride=module.stride, dilation=module.dilation)
                        fea_in_ = fea_in_.permute(0, 2, 1)  # [1, num_patch, c_in*kernel_size]
                        fea_in_ = fea_in_.reshape(-1, fea_in_.shape[-1])  # [num_patch, c_in*kernel_size]
                        fea_in_ = torch.mean(fea_in_, 0, True)
                        self.get_buffer(concat_param_names([module_name, 'weight']), prefix='feat_in').copy_(fea_in_)
                    if concat_param_names([module_name, 'bias']) in self.param_names_require_owm:
                        mean_y = mean_y.view(1, module.out_channels, -1)  # [1, out_c, img_size]
                        mean_y = torch.mean(mean_y, 2)  # [1, out_c]
                        fea_in_ = mean_y - torch.unsqueeze(module.bias.detach(), 0)
                        self.get_buffer(concat_param_names([module_name, 'bias']), prefix='feat_in').copy_(fea_in_)
                elif isinstance(module, torch.nn.BatchNorm1d) \
                        or isinstance(module, torch.nn.BatchNorm2d) \
                        or isinstance(module, torch.nn.BatchNorm3d):
                    bias = torch.unsqueeze(module.bias.detach(), 0)
                    weight = torch.unsqueeze(module.weight.detach(), 0)
                    for _ in range(mean_y.ndim - bias.ndim):
                        bias = torch.unsqueeze(bias, -1)
                        weight = torch.unsqueeze(weight, -1)
                    fea_in_bias = mean_y.detach() - bias
                    if concat_param_names([module_name, 'weight']) in self.param_names_require_owm:
                        fea_in_weight = torch.mean(fea_in_bias / (weight + 1e-8), 0, True)
                        while fea_in_weight.ndim > 2:
                            fea_in_weight = torch.mean(fea_in_weight, -1, False)
                        self.get_buffer(concat_param_names([module_name, 'weight']), prefix='feat_in') \
                            .copy_(fea_in_weight)
                    if concat_param_names([module_name, 'bias']) in self.param_names_require_owm:
                        fea_in_bias = torch.mean(fea_in_bias, 0, True)
                        while fea_in_bias.ndim > 2:
                            fea_in_bias = torch.mean(fea_in_bias, -1, False)
                        self.get_buffer(concat_param_names([module_name, 'bias']), prefix='feat_in') \
                            .copy_(fea_in_bias)
                elif isinstance(module, torch.nn.Linear):
                    if concat_param_names([module_name, 'weight']) in self.param_names_require_owm:
                        self.get_buffer(concat_param_names([module_name, 'weight']), prefix='feat_in') \
                            .copy_(mean_x)
                    if concat_param_names([module_name, 'bias']) in self.param_names_require_owm:
                        bias = torch.unsqueeze(module.bias.detach(), 0)
                        fea_in_ = mean_y - bias
                        self.get_buffer(concat_param_names([module_name, 'weight']), prefix='feat_in') \
                            .copy_(fea_in_)
                else:
                    raise RuntimeError('Module %s is not defined in OWM' % module_name)

        return hook

    def training_step(self, batch, batch_idx, **kwargs) -> torch.Tensor:
        inputs, labels = batch
        outputs = self.forward(inputs)
        standard_loss = self.criterion(outputs, labels)

        num_train_batches = self.trainer.num_training_batches
        cur_task_epoch = self.num_task_epochs - (self.trainer.max_epochs - self.trainer.current_epoch)
        lam_e = (batch_idx / num_train_batches + cur_task_epoch) / self.num_task_epochs

        self.step_alpha = dict()
        for i, param_name in enumerate(self.param_names_require_owm):
            self.step_alpha[param_name] = self.alpha * self.a_lambda[i] ** lam_e

        self.shared_step_log('train', batch_idx, inputs, outputs, labels)
        return standard_loss

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs) -> None:
        super().backward(loss, optimizer, optimizer_idx, *args, **kwargs)

        named_params = dict(self.model.named_parameters())
        with torch.no_grad():
            for param_name in self.param_names_require_owm:
                x = self.get_buffer(param_name, prefix='feat_in')
                p = self.get_buffer(param_name, prefix='p')
                step_alpha = self.step_alpha[param_name]

                param = named_params[param_name]
                if param.ndim == 1:  # bn, fc...
                    k = torch.mm(p, torch.t(x))
                    p.sub_(torch.mm(k, torch.t(k)) / (step_alpha + torch.mm(x, k)))
                    param.grad.copy_(torch.mm(param.grad.view(1, -1), torch.t(p)).view(param.shape))
                else:  # conv, conv_t
                    for sample in x:
                        sample = sample.unsqueeze(0)
                        k = torch.mm(p, torch.t(sample))
                        p.sub_(torch.mm(k, torch.t(k)) / (step_alpha + torch.mm(sample, k)))
                    param.grad.copy_(torch.mm(param.grad.view(param.shape[0], -1), torch.t(p)).view(param.shape))

    def load_state_dict(self, state_dict, strict: bool = True):
        named_buffers = dict(self.named_buffers(recurse=False))
        ps_unwanted = list()
        ps_newly_added = list()
        for k, v in state_dict.items():
            if k.startswith('p/') and k not in named_buffers.keys():
                ps_unwanted.append(k)
        for k, v in named_buffers.items():
            if k.startswith('p/') and k not in state_dict:
                ps_newly_added.append(k)
        for k in ps_unwanted:
            state_dict.pop(k)
        for k in ps_newly_added:
            state_dict[k] = self.__getattr__(k)
        super().load_state_dict(state_dict, strict)


def concat_param_names(name_list: List[str], param_name_sep='.'):
    return param_name_sep.join(name_list)


def get_param_last_name(param_name: str, param_name_sep='.', param_name_lib: List[str] = None):
    if param_name_lib is None:
        param_name_lib = ['weight', 'bias']
    name = param_name.split(param_name_sep)[-1]
    if name not in param_name_lib:
        raise ValueError('Unrecognized param name: %s' % name)
    return name


def get_param_direct_module_father_name(param_name: str, param_name_sep='.', param_name_lib: List[str] = None):
    name_list = param_name.split(param_name_sep)
    if param_name_lib is None:
        param_name_lib = ['weight', 'bias']
    while name_list[-1] in param_name_lib:
        name_list.pop(-1)
    return param_name_sep.join(name_list)
