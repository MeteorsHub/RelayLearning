from torch import nn


def get_norm(norm_type: str, num_features=None, spatial_dim: int = None, **kwargs):
    if spatial_dim is not None:
        assert 1 <= spatial_dim <= 3

    if norm_type == 'none':
        return nn.Identity(**kwargs)
    if norm_type == 'bn':
        if spatial_dim is None or spatial_dim == 2:
            return nn.BatchNorm2d(num_features=num_features, **kwargs)
        if spatial_dim == 1:
            return nn.BatchNorm1d(num_features=num_features, **kwargs)
        if spatial_dim == 3:
            return nn.BatchNorm3d(num_features=num_features, **kwargs)
    if norm_type == 'ln':
        return nn.LayerNorm(**kwargs)
    if norm_type == 'in':
        if spatial_dim is None or spatial_dim == 2:
            return nn.InstanceNorm2d(num_features=num_features, **kwargs)
        if spatial_dim == 1:
            return nn.InstanceNorm1d(num_features=num_features, **kwargs)
        if spatial_dim == 3:
            return nn.InstanceNorm3d(num_features=num_features, **kwargs)
    if norm_type == 'gn':
        return nn.GroupNorm(num_channels=num_features, **kwargs)
    raise AttributeError('Unsupported norm type %s' % norm_type)


def get_nonlinear(nl_type, inplace=True, **kwargs):
    if nl_type == 'none':
        return nn.Identity(**kwargs)
    if nl_type == 'relu':
        return nn.ReLU(inplace=inplace)
    if nl_type == 'leaky_relu':
        return nn.LeakyReLU(inplace=inplace, **kwargs)
    if nl_type == 'celu':
        return nn.CELU(inplace=inplace, **kwargs)
    if nl_type == 'gelu':
        return nn.GELU()
    if nl_type == 'sigmoid':
        return nn.Sigmoid()
    if nl_type == 'softmax':
        if 'dim' not in kwargs:
            return nn.Softmax(dim=1)
        else:
            return nn.Softmax(**kwargs)
    if nl_type == 'tanh':
        return nn.Tanh()
    raise AttributeError('Unsupported nonlinear type %s' % nl_type)
