import torch

from models.basic import DeepLearningRelayModel
from strategies.basic import MultiSiteLearningStrategy


class Relay(MultiSiteLearningStrategy):
    name = 'relay'
    model: DeepLearningRelayModel

    def __init__(self, model: DeepLearningRelayModel, config: dict, **kwargs):
        super().__init__(model, config, **kwargs)

    def configure_optimizers(self):
        return self.model.configure_optimizers()

    def training_step(self, batch, batch_idx, optimizer_idx=None) -> torch.Tensor:
        model_loss = self.model.training_step(batch, batch_idx, optimizer_idx)
        return model_loss

    def on_train_epoch_start(self) -> None:
        self.model.on_train_epoch_start()
        super().on_train_epoch_start()

    def on_train_batch_start(self, batch, batch_idx: int, dataloader_idx: int) -> None:
        self.model.on_train_batch_start(batch, batch_idx, dataloader_idx)
        super().on_train_batch_start(batch, batch_idx, dataloader_idx)

    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        self.model.on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)
        super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def validation_step(self, batch, batch_idx) -> None:
        self.model.validation_step(batch, batch_idx)
        # super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        self.model.test_step(batch, batch_idx, dataloader_idx)
        # super().test_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def on_train_start(self):
        if self.previous_task_id > 0 and self.task_id == self.previous_task_id + 1:
            self.model.start_new_task()
        self.model.on_train_start()
        super().on_train_start()

    def on_save_checkpoint(self, checkpoint) -> None:
        self.model.on_save_checkpoint(checkpoint)
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint) -> None:
        self.model.on_load_checkpoint(checkpoint)
        super().on_load_checkpoint(checkpoint)
