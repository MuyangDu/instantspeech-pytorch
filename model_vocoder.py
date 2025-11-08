# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm
from config import Config

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class ResBlock1(torch.nn.Module):
    def __init__(self, config, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.config = config

        Conv1d = torch.nn.Conv1d

        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            if self.config.act_func == "relu":
                xt = F.relu(x)
            else:
                xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            if self.config.act_func == "relu":
                xt = F.relu(xt)
            else:
                xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
            
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, config, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.config = config

        Conv1d = torch.nn.Conv1d
        
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            if self.config.act_func == "relu":
                xt = F.relu(x)
            else:
                xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
            
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        pad_tensor = torch.zeros((input.shape[0], input.shape[1], self.__padding), device=input.device)
        input = torch.cat((pad_tensor, input), dim=2)

        return super(CausalConv1d, self).forward(input)
        # return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class CasualResBlock1(torch.nn.Module):
    def __init__(self, config, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(CasualResBlock1, self).__init__()
        self.config = config
        
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.convs1 = nn.ModuleList([
            weight_norm(CausalConv1d(channels, channels, kernel_size, stride=1, dilation=dilation[0])),
            weight_norm(CausalConv1d(channels, channels, kernel_size, stride=1, dilation=dilation[1])),
            weight_norm(CausalConv1d(channels, channels, kernel_size, stride=1, dilation=dilation[2]))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(CausalConv1d(channels, channels, kernel_size, stride=1, dilation=1)),
            weight_norm(CausalConv1d(channels, channels, kernel_size, stride=1, dilation=1)),
            weight_norm(CausalConv1d(channels, channels, kernel_size, stride=1, dilation=1))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x, states=None, state_index=0, new_states=None):
        index = 0
        
        for c1, c2 in zip(self.convs1, self.convs2):
            if self.config.act_func == "relu":
                xt = F.relu(x)
            else:
                xt = F.leaky_relu(x, LRELU_SLOPE)
                
            if states is None:
                xt = c1(xt)
            else:
                ps = states[state_index]
                state_index += 1
                ns = xt[:, :, -((self.kernel_size-1) * self.dilation[index]):]
                xt = torch.cat((ps, xt), dim=2)
                xt = c1(xt)
                xt = xt[:, :, (self.kernel_size-1) * self.dilation[index]:]
                new_states.append(ns)
                index += 1

            if self.config.act_func == "relu":
                xt = F.relu(xt)
            else:
                xt = F.leaky_relu(xt, LRELU_SLOPE)
                
            if states is None:
                xt = c2(xt)
            else:
                ps = states[state_index]
                state_index += 1
                ns = xt[:, :, -(self.kernel_size-1):]
                xt = torch.cat((ps, xt), dim=2)
                xt = c2(xt)
                xt = xt[:, :, self.kernel_size-1:]
                new_states.append(ns)
            
            x = xt + x
            
        if states is not None:
            return x, state_index
            
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class CasualResBlock2(torch.nn.Module):
    def __init__(self, config, channels, kernel_size=3, dilation=(1, 3)):
        super(CasualResBlock2, self).__init__()
        self.config = config
        
        self.convs = nn.ModuleList([
            weight_norm(CausalConv1d(channels, channels, kernel_size, stride=1, dilation=dilation[0])),
            weight_norm(CausalConv1d(channels, channels, kernel_size, stride=1, dilation=dilation[1]))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            if self.config.act_func == "relu":
                xt = F.relu(x)
            else:
                xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
            
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class UpsampleBlock(torch.nn.Module):
    def __init__(self, config, in_channels, out_channels, upsample_rate, upsample_kernel_size):
        super(UpsampleBlock, self).__init__()
        self.config = config
        
        self.upsample_kernel_size = upsample_kernel_size
        
        self.up = torch.nn.Upsample(
            scale_factor=upsample_rate, mode='nearest')
        
        if config.upsample_type == "interconv":
            self.conv = weight_norm(CausalConv1d(in_channels, out_channels, upsample_kernel_size, stride=1))
        else:
            raise NotImplementedError
        
        self.conv.apply(init_weights)
    
    def forward(self, x, states=None, state_index=0, new_states=None):
        x = self.up(x)
        
        if states is None:
            x = self.conv(x)
            return x
        else:
            ps = states[state_index]
            state_index += 1
            ns = x[:, :, -(self.upsample_kernel_size-1):]
            x = torch.cat((ps, x), dim=2)
            x = self.conv(x)
            x = x[:, :, self.upsample_kernel_size-1:]
            new_states.append(ns)
            return x, state_index
    

class Generator(torch.nn.Module):
    def __init__(self, config: Config):
        super(Generator, self).__init__()
        self.config = config
        
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        
        if config.casual_generator:
            self.conv_pre = weight_norm(CausalConv1d(config.n_mel_channels, config.upsample_initial_channel, 7, 1))
            resblock = CasualResBlock1 if config.resblock == '1' else CasualResBlock2
        else:
            self.conv_pre = weight_norm(Conv1d(config.n_mel_channels, config.upsample_initial_channel, 7, 1, padding=3))
            resblock = ResBlock1 if config.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            if config.upsample_type == "interconv":
                self.ups.append(UpsampleBlock(config, in_channels=config.upsample_initial_channel//(2**i), 
                                                          out_channels=config.upsample_initial_channel//(2**(i+1)), 
                                                          upsample_rate=u, upsample_kernel_size=k))
            elif config.upsample_type == "convtranspose":
                self.ups.append(weight_norm(torch.nn.ConvTranspose1d(config.upsample_initial_channel//(2**i), 
                                                                     config.upsample_initial_channel//(2**(i+1)),
                                                                     k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel//(2**(i+1))
            
            for j, (k, d) in enumerate(zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)):
                    self.resblocks.append(resblock(config, ch, k, d))
        
        if config.casual_generator:
            self.conv_post = weight_norm(CausalConv1d(ch, 1, 7, 1))
        else:
            self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
    

    def forward(self, x, states=None):
        if states is not None:
            new_states = []
            state_index = 0
        
        if states is None:
            x = self.conv_pre(x)
        else:
            ps = states[state_index]
            state_index += 1
            ns = x[:, :, -6:]
            x = torch.cat((ps, x), dim=2)
            x = self.conv_pre(x)
            x = x[:, :, 6:]
            new_states.append(ns)
            
        for i in range(self.num_upsamples):
            if self.config.act_func == "relu":
                x = F.relu(x)
            else:
                x = F.leaky_relu(x, LRELU_SLOPE)
            
            if states is None:
                x = self.ups[i](x)
            else:
                x, state_index = self.ups[i](x, states, state_index, new_states)
            
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    if states is None:
                        xs = self.resblocks[i*self.num_kernels+j](x)
                    else:
                        xs, state_index = self.resblocks[i*self.num_kernels+j](x, states, state_index, new_states)
                else:
                    if states is None:
                        xs += self.resblocks[i*self.num_kernels+j](x)
                    else:
                        res, state_index = self.resblocks[i*self.num_kernels+j](x, states, state_index, new_states)
                        xs += res
            
            x = xs / self.num_kernels
        
        if self.config.act_func == "relu":
            x = F.relu(x)
        else:
            x = F.leaky_relu(x)
        
        if states is None:
            x = self.conv_post(x)
        else:
            ps = states[state_index]
            state_index += 1
            ns = x[:, :, -6:]
            x = torch.cat((ps, x), dim=2)
            x = self.conv_post(x)
            x = x[:, :, 6:]
            new_states.append(ns)
        
        x = torch.tanh(x)
        
        if states is not None:
            return x, new_states

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        # for l in self.ups:
        #     remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)