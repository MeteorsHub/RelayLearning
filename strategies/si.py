import torch

from models.basic import DeepLearningModel
from strategies.basic import MultiSiteLearningStrategy


class SI(MultiSiteLearningStrategy):
    name = 'si'

    def __init__(self, model: DeepLearningModel, config: dict, c=0.5, xi=1.0, **kwargs):
        super().__init__(model, config, **kwargs)

        self.save_hyperparameters('c', 'xi')
        self.c = c
        self.xi = xi

        self.register_buffer('checkpoint_params', torch.zeros_like(
            self.model.get_params(skip_untrainable_params=True), device=self.device))
        self.register_buffer('small_omega', torch.zeros_like(
            self.model.get_params(skip_untrainable_params=True), device=self.device))
        self.register_buffer('big_omega', torch.zeros_like(
            self.model.get_params(skip_untrainable_params=True), device=self.device))

    def training_step(self, batch, batch_idx, **kwargs) -> torch.Tensor:
        inputs, labels = batch
        outputs = self.forward(inputs)
        standard_loss = self.criterion(outputs, labels)
        penalty = self.penalty()
        loss = standard_loss + self.c * penalty

        self.shared_step_log('train', batch_idx, inputs, outputs, labels)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        self.small_omega.add_(self.get_current_lr() * self.model.get_grads(untrainable_params='skip').detach() ** 2)
        super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def penalty(self):
        if self.task_id == 1:
            return torch.tensor(0.0, device=self.device)
        else:
            penalty = (self.big_omega * (
                    (self.model.get_params(skip_untrainable_params=True) - self.checkpoint_params.detach()) ** 2)).sum()
            return penalty

    def on_train_end(self):
        with torch.no_grad():
            self.small_omega.clamp_(-1e8, 1e8)
            self.big_omega.add_(self.small_omega / (
                    (self.model.get_params(skip_untrainable_params=True).detach() -
                     self.checkpoint_params.detach()) ** 2 + self.xi))
            self.checkpoint_params.copy_(self.model.get_params(skip_untrainable_params=True))
            self.small_omega.zero_()
        super().on_train_end()
