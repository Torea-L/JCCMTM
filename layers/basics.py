import torch
from torch import nn
from torch import Tensor

def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    elif activation.lower() == "elu": return nn.ELU()
    elif activation.lower() == "leaky_relu": return nn.LeakyReLU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')

def get_norm_fn(normalization, norm_shape):
    if callable(normalization): return normalization
    # elif "my" in normalization.lower():
    #     return my_Layernorm(norm_shape)
    elif "batch" in normalization.lower():
        return nn.BatchNorm1d(norm_shape)
    elif "layer" in normalization.lower():
        return nn.LayerNorm(norm_shape)
    raise ValueError(f'{normalization} is not available. You can use "BatchNorm", "LayerNorm", or a callable')

## decomposition
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x:torch.Tensor):
        """
        Input:
            x: [Batch, Input length, Channel] or [Input length, Channel]
        Return:
            x: [Batch, Input length, Channel] or [Input length, Channel]
        """
        if len(x.shape)==3:
            # padding on the both ends of time series
            front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            x = torch.cat([front, x, end], dim=1)
            x = self.avg(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        elif len(x.shape)==2:
            front = x[0:1, :].repeat((self.kernel_size - 1) // 2, 1)
            end = x[-1:, :].repeat((self.kernel_size - 1) // 2, 1)
            x = torch.cat([front, x, end], dim=0)
            x = self.avg(x.permute(1, 0))
            x = x.permute(1, 0)
        else:
            raise Exception(
                'Unsupported data shape: {}, [bsz, seq_len, patch_len] or [series_len, n_vars]'.format(x.shape))
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size, stride=1):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=stride)

    def forward(self, x):
        """
        Input:
            x: [Batch, Input length, Channel]
        Return:
            res: [Batch, Input length, Channel]
            moving_mean : [Batch, Input length, Channel]
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    

class raw_series_decomp(nn.Module):
    """
    Series decomposition block used in data loading stage
    """
    def __init__(self, kernel_size, stride=1):
        super(raw_series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=stride)

    def forward(self, x:torch.Tensor):
        # x : [series_len, n_vars]
        moving_mean = self.moving_avg(x)                        # moving_mean : [series_len, n_vars]
        res = x - moving_mean                                   # res : [series_len, n_vars]
        return res.numpy(), moving_mean.numpy()
