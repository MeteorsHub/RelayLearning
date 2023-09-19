from datasets.basic import MultiSiteDataset
from datasets.fundus import FundusDataset
from datasets.mediastinum_tumor import MediastinumTumorDataset
from datasets.midline import MidlineDataset

__all__ = ['all_datasets', 'get_dataset']

all_datasets = {
    'mediastinum-tumor': MediastinumTumorDataset,
    'fundus': FundusDataset,
    'midline': MidlineDataset,
}


def get_dataset(name: str, train: bool, task: str = None, num_classes: int = None, task_id=None, task_name=None,
                **kwargs) -> MultiSiteDataset:
    assert name in all_datasets, 'dataset %s does not exist' % name
    return all_datasets[name](train=train, task=task, num_classes=num_classes,
                              task_id=task_id, task_name=task_name, **kwargs)
