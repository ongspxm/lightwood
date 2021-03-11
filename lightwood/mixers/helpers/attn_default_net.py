import math
import torch
from functools import reduce

from lightwood.config.config import CONFIG
from lightwood.mixers.helpers.shapes import *
from lightwood.mixers.helpers.plinear import PLinear
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.helpers.device import get_devices
from lightwood.logger import log


class AttnDefaultNet(torch.nn.Module):

    def __init__(self, dynamic_parameters,
                 input_size=None,
                 output_size=None,
                 nr_outputs=None,
                 shape=None,
                 max_params=3e5,
                 dropout=None,
                 pretrained_net=None):
        self.input_size = input_size
        self.output_size = output_size
        self.nr_outputs = nr_outputs
        self.max_variance = None
        self.device, self.available_devices = get_devices()
        self.dynamic_parameters = dynamic_parameters

        super(AttnDefaultNet, self).__init__()

        if shape is None and pretrained_net is None:
            # default shape, inflated, yields 240k params if input has 200 components
            hidden_size = max([self.input_size*2, self.output_size*2, 400])
            shape = [self.input_size,
                     hidden_size,
                     self.output_size]

            # if NN is too big, we do not inflate and aim for max_params instead
            if reduce(lambda x, y: x*y, shape) > max_params:
                hidden_size = math.floor(max_params/(self.input_size*self.output_size))

                if hidden_size > self.output_size:
                    shape = [self.input_size, hidden_size, self.output_size]
                else:
                    shape = [self.input_size, self.output_size]

        if pretrained_net is None:
            log.info(f'Building network of shape: {shape}')

            layers = []
            for ind in range(len(shape) - 1):
                if (dropout is not None) and (0 < ind < len(shape)):
                    layers.append(torch.nn.Dropout(p=dropout))
                if ind < len(shape) - 2:
                    layers.append(LinAttnBlock(shape[ind], shape[ind+1]))
                else:
                    layers.append(LinAttnBlock(shape[ind], shape[ind+1], activation=False))

            self.net = torch.nn.Sequential(*layers)

        else:
            self.net = pretrained_net
            for layer in self.net:
                # TODO: this might not work okay
                if isinstance(layer, torch.nn.Linear):
                    if self.input_size is None:
                        self.input_size = layer.in_features
                    self.output_size = layer.out_features

        self.net.to(self.device)

        if self.available_devices > 1:
            self.net = torch.nn.DataParallel(self.net)

    def to(self, device=None, available_devices=None):
        if device is None or available_devices is None:
            device, available_devices = get_devices()

        self.net = self.net.to(device)

        available_devices = 1
        if 'cuda' in str(device):
            available_devices = torch.cuda.device_count()

        if self.available_devices > 1:
            self.net = torch.nn.DataParallel(self.net)

        self.device = device
        self.available_devices = available_devices

        return self

    def forward(self, input):
        """
        In this particular model, we just need to forward the network defined in setup, with our input
        :param input: a pytorch tensor with the input data of a batch
        :return: output of the network
        """
        with LightwoodAutocast():
            output = self.net(input)

        return output


class LinAttnBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=1, activation=True):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.attn = torch.nn.MultiheadAttention(out_dim, n_heads)
        self.activation = None if not activation else torch.nn.SELU()

    def residual_attn(self, x):
        """ :param x: shape (B, n_feats) """
        x = x.unsqueeze(dim=0)
        res = x
        x_att, _ = self.attn(x, x, x)
        x = x_att.squeeze(dim=0)
        res = res.squeeze(dim=0)
        return x + res

    def forward(self, x):
        out = self.linear(x)
        out = self.residual_attn(out)
        if self.activation is not None:
            return self.activation(out)
        else:
            return out