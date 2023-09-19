from models.basic import DeepLearningGenerator, DeepLearningModel, DeepLearningRelayModel
from models.doublegan import DoubleGAN
from models.owm_net import OWMNet
from models.resnet import *
from models.resnet_s import *
from models.unet import UNet

__all__ = ['all_models', 'DeepLearningModel', 'DeepLearningGenerator', 'DeepLearningRelayModel']

all_models = {
    'U-Net': UNet,

    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,

    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet1202': resnet1202,

    'OWM-Net': OWMNet,

    'doublegan': DoubleGAN,
}


def get_model(name: str, **kwargs) -> DeepLearningModel:
    assert name in all_models, 'model %s does not exist' % name
    return all_models[name](**kwargs)
