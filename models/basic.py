from abc import ABC

import torch
from pytorch_lightning import LightningModule
from torch import Tensor


class DeepLearningModel(LightningModule, ABC):
    task: str = None
    _base_strategy = None

    def configure_base_strategy(self, base_strategy):
        # in order to not re-register this base_strategy module in pytorch
        self._base_strategy = [base_strategy]

    @property
    def base_strategy(self):
        if self._base_strategy is None:
            return None
        return self._base_strategy[0]

    def configure_optimizers(self):
        return None

    def forward(self, *args, **kwargs):
        """
        This method return model output
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx=None, **kwargs):
        """
        This method return loss.
        """
        return None

    def validation_step(self, batch, batch_idx, **kwargs):
        return None

    def test_step(self, batch, batch_idx, dataloader_idx=None, **kwargs):
        return None

    def get_params(self, skip_untrainable_params=False) -> torch.Tensor:
        params = []
        for pp in list(self.parameters()):
            if skip_untrainable_params and not pp.requires_grad:
                continue
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor, skip_untrainable_params=False) -> None:
        assert new_params.size() == self.get_params(skip_untrainable_params).size()
        progress = 0
        for pp in list(self.parameters()):
            if skip_untrainable_params and not pp.requires_grad:
                continue
            cand_params = new_params[progress: progress + torch.tensor(pp.size(), device=pp.device).prod()]. \
                view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self, untrainable_params='zero') -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        assert untrainable_params in ['skip', 'zero']
        grads = []
        for pp in list(self.parameters()):
            if not pp.requires_grad:
                if untrainable_params == 'zero':
                    grads.append(torch.zeros(pp.shape, device=pp.device).view(-1))
                if untrainable_params == 'skip':
                    continue
                else:
                    raise AttributeError
            else:
                grads.append(pp.grad.view(-1))
        return torch.cat(grads)


class DeepLearningGenerator(DeepLearningModel, ABC):
    trained_c_ids = []

    def sample(self, batch_size, device=None, **kwargs):
        raise NotImplementedError


class DeepLearningRelayModel(DeepLearningModel, ABC):
    def forward(self, *args, **kwargs):
        """
        This method return the main task output
        """
        raise NotImplementedError

    def start_new_task(self):
        raise NotImplementedError

    def replay(self, num_samples):
        raise NotImplementedError


class DeepLearningBackbone(DeepLearningModel, ABC):
    num_feature_layers: int = None  # extract this number of feature layers
    num_feature_channels: list = None  # for all layers
    task = 'classification'

    sub_nets = {
        'feature_net': None,
        'classifier': None
    }

    def get_feature_net(self):
        return self.sub_nets['feature_net']

    def get_classifier(self):
        return self.sub_nets['classifier']

    def forward_features(self, x: Tensor) -> list:
        raise NotImplementedError
