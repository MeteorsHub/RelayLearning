import csv
import datetime
import io
import logging
import os
from typing import Dict, Optional, Union

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

log = logging.getLogger(__name__)


class ExperimentWriter(object):
    r"""
    Experiment writer for ExpLogger.

    Currently supports to log hyperparameters and metrics in YAML and CSV
    format, respectively.

    Args:
        log_dir: Directory for the experiment logs
    """
    NAME_EXP_INFO_FILE = 'experiment_info.csv'

    def __init__(self, log_dir: str) -> None:
        self.exp_info = []

        self.log_dir = log_dir
        if os.path.exists(self.log_dir) and os.listdir(self.log_dir):
            rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
        os.makedirs(self.log_dir, exist_ok=True)

        self.exp_info_file_path = os.path.join(self.log_dir, self.NAME_EXP_INFO_FILE)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d#%H-%M-%S')

    def log_info_dict(self, info_dict: Dict[str, Union[torch.Tensor, float, str]], step: Optional[int] = None) -> None:
        def _handle_value(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy().tolist()
            return value

        if step is None:
            step = len(self.exp_info)
        exp_info = {'step': step}
        for k, v in info_dict.items():
            exp_info[k] = _handle_value(v)
        for i in range(len(self.exp_info)):
            if self.exp_info[i]['step'] == step:
                self.exp_info[i].update(exp_info)
                return
        self.exp_info.append(exp_info)

    def save(self) -> None:
        if not self.exp_info:
            return

        last_m = {}
        for m in self.exp_info:
            last_m.update(m)
        exp_info_keys = list(last_m.keys())

        file = self.exp_info_file_path.replace(self.NAME_EXP_INFO_FILE, self.time_str + '_' + self.NAME_EXP_INFO_FILE)
        with io.open(file, 'w', newline='') as f:
            self.writer = csv.DictWriter(f, fieldnames=exp_info_keys)
            self.writer.writeheader()
            self.writer.writerows(self.exp_info)


class ExpLogger(LightningLoggerBase):
    def __init__(
            self,
            save_dir: str,
            name: Optional[str] = "default",
            version: Optional[Union[int, str]] = None,
            prefix: str = '',
    ):
        super().__init__()
        self._save_dir = save_dir
        self._name = name or ''
        self._version = version
        self._prefix = prefix
        self._experiment = None

    @property
    def root_dir(self) -> str:
        """
        Parent directory for all checkpoint subdirectories.
        If the experiment name parameter is ``None`` or the empty string, no experiment subdirectory is used
        and the checkpoint will be saved in "save_dir/version_dir"
        """
        if not self.name:
            return self.save_dir
        return os.path.join(self.save_dir, self.name)

    @property
    def log_dir(self) -> str:
        """
        The log directory for this run. By default, it is named
        ``'version_${self.version}'`` but it can be overridden by passing a string value
        for the constructor's version parameter instead of ``None`` or an int.
        """
        # create a pseudo standard path ala test-tube
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        return log_dir

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir

    @property
    @rank_zero_experiment
    def experiment(self) -> ExperimentWriter:
        r"""

        Actual ExperimentWriter object. To use ExperimentWriter features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment:
            return self._experiment

        os.makedirs(self.root_dir, exist_ok=True)
        self._experiment = ExperimentWriter(log_dir=self.log_dir)
        return self._experiment

    @rank_zero_only
    def save(self) -> None:
        super().save()
        self.experiment.save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.save()

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = os.path.join(self._save_dir, self.name)

        if not os.path.isdir(root_dir):
            log.warning('Missing logger folder: %s', root_dir)
            return 0

        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    @rank_zero_only
    def log_hyperparams(self, params) -> None:
        pass

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass
