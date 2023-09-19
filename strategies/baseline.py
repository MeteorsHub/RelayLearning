import torch

from strategies.basic import MultiSiteLearningStrategy


class Baseline(MultiSiteLearningStrategy):
    name: str = 'baseline'

    def training_step(self, batch, batch_idx, **kwargs) -> torch.Tensor:
        inputs, labels = batch

        outputs = self.forward(inputs)
        standard_loss = self.criterion(outputs, labels)

        self.shared_step_log('train', batch_idx, inputs, outputs, labels)
        return standard_loss
