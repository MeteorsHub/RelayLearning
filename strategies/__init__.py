from models.basic import DeepLearningModel
from strategies.baseline import Baseline
from strategies.basic import MultiSiteLearningStrategy
from strategies.owm import OWM
from strategies.relay import Relay
from strategies.si import SI

__all__ = ['all_strategies', 'get_strategy']

all_strategies = {
    'baseline': Baseline,
    'si': SI,
    'owm': OWM,
    'relay': Relay
}


def get_strategy(name: str, model: DeepLearningModel, task_conf: dict, **kwargs) -> MultiSiteLearningStrategy:
    assert name in all_strategies, 'strategy %s does not exist' % name
    return all_strategies[name](model=model, config=task_conf, **kwargs)
