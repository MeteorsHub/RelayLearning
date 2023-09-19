import logging
from typing import Union

from imgaug.augmenters.meta import Augmenter


class MultiSiteDataset:
    task: str = None
    num_classes: int = None

    task_id: int = None
    task_name: str = None

    aug_pipe: Augmenter = None
    num_samples: int = None

    def __init__(self, train: bool, task: str = None, num_classes: int = None, task_id=None, task_name=None):
        super().__init__()
        self.train = train
        if task is not None:
            self.task = task
        if num_classes is not None:
            self.num_classes = num_classes
        self.task_id = task_id
        self.task_name = task_name
        self.logger = logging.getLogger('MultiSiteDataset')

    def get_subject_item_id(self, item: int) -> (Union[int, str], Union[int, str]):
        # determine index in __getitem__ belongs to which subject id and slice_id
        raise NotImplementedError
