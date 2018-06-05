import torch
import torch.nn as nn
import numpy as np
from torch import autograd
from skimage.io import imread
import matplotlib.pyplot as plt

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation_fn=None, use_bn=False):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.activ = activation_fn
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activ is not None:
            x = self.activ(x)
        return x

class Generator(nn.Module):
    def __init__(
        self,
        input_channel,
        z_num=64,
        repeat_num = int(np.log2(256) - 2),
        hidden_num = 128,
        activation_fn=torch.nn.functional.elu,
        min_fea_map_H=8,
        n_channels_pose=3,
        *args, **kwargs
    ):
        super(Generator, self).__init__(*args, **kwargs)
        assert min_fea_map_H * 2 ** (repeat_num - 1) == 128
        self.repeat_num = repeat_num
        self.min_fea_map_H = min_fea_map_H
        self.hidden_num = hidden_num
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder.add_module(
            'conv_0_0', conv_block(input_channel + n_channels_pose, hidden_num, 3, 1, activation_fn)
        )
        
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            self.encoder.add_module(
                'conv_{}_{}'.format(idx + 1, 0), conv_block(channel_num, channel_num, 3, 1, activation_fn)
            )
            self.encoder.add_module(
                'conv_{}_{}'.format(idx + 1, 1), conv_block(channel_num, channel_num, 3, 1, activation_fn)
            )
            if idx < repeat_num - 1:
                self.encoder.add_module(
                    'down_{}'.format(idx + 1), conv_block(channel_num, channel_num + hidden_num, 3, 2, activation_fn)
                )
                
        # Bottleneck
        self.bottleneck = nn.Linear(int(np.prod([min_fea_map_H, min_fea_map_H, channel_num])), z_num)
        
        self.decoder = nn.ModuleList()
        
        self.decoder.add_module('fc0', nn.Linear(z_num, int(np.prod([min_fea_map_H, min_fea_map_H, hidden_num]))))
        
        for idx in range(repeat_num):
            channel_num = hidden_num * (repeat_num + 1) if idx == 0 else 2 * hidden_num*(repeat_num-idx)
            #hidden_num * (repeat_num + 1) if idx < repeat_num - 1 else 2 * hidden_num
            self.decoder.add_module(
                'conv_{}_{}'.format(idx, 0), conv_block(channel_num, channel_num, 3, 1, activation_fn)
            )
            self.decoder.add_module(
                'conv_{}_{}'.format(idx, 1), conv_block(channel_num, channel_num, 3, 1, activation_fn)
            )
            if idx < repeat_num - 1:
                self.decoder.add_module(
                    'up_{}'.format(idx), nn.Upsample(scale_factor=2, mode='nearest')
                )
                self.decoder.add_module(
                    'up_conv_{}'.format(idx), conv_block(channel_num, hidden_num*(repeat_num-idx-1), 1, 1, activation_fn)
                )
                
        self.head = conv_block(2 * hidden_num, input_channel, 3, 1, activation_fn=None)
    
    def forward(self, x):
        self.encoder_list = []
        # Encode
        x = self.encoder.conv_0_0(x)
        for idx in range(self.repeat_num):
            res = x
            x = self.encoder.__getattr__('conv_{}_{}'.format(idx + 1, 0))(x)
            x = self.encoder.__getattr__('conv_{}_{}'.format(idx + 1, 1))(x)
            x = res + x
            self.encoder_list.append(x)
            if idx < self.repeat_num - 1:
                x = self.encoder.__getattr__('down_{}'.format(idx + 1))(x)
        # Bottleneck
        z = x = self.bottleneck(torch.reshape(x, (x.shape[0], -1)))
        # Decode
        x = self.decoder.fc0(x)
        x = torch.reshape(x, (x.shape[0], self.hidden_num, self.min_fea_map_H, self.min_fea_map_H))
        for idx in range(self.repeat_num):
            x = torch.cat([x, self.encoder_list[self.repeat_num-1-idx]], dim=1)
            res = x
            x = self.decoder.__getattr__('conv_{}_{}'.format(idx, 0))(x)
            x = self.decoder.__getattr__('conv_{}_{}'.format(idx, 1))(x)
            x = x + res
            if idx < self.repeat_num - 1:
                x = self.decoder.__getattr__('up_{}'.format(idx))(x)
                x = self.decoder.__getattr__('up_conv_{}'.format(idx))(x)
                
        out = self.head(x)
        return out, z
    
class Refiner(nn.Module):
    def __init__(
        self,
        input_channel,
        repeat_num = int(np.log2(256)) - 2,
        hidden_num = 128,
        activation_fn=torch.nn.functional.elu,
        min_fea_map_H=8,
        noise_dim=0,
        device=torch.device('cpu:0'),
        *args, **kwargs
    ):
        super(Refiner, self).__init__(*args, **kwargs)
        assert min_fea_map_H * 2 ** (repeat_num - 1) == 128
        self.repeat_num = repeat_num
        self.min_fea_map_H = min_fea_map_H
        self.hidden_num = hidden_num
        self.noise_dim = noise_dim
        self.device = device
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder.add_module(
            'conv_0_0', conv_block(2 * input_channel, hidden_num, 3, 1, activation_fn)
        )
        
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            self.encoder.add_module(
                'conv_{}_{}'.format(idx + 1, 0), conv_block(max(channel_num - hidden_num, hidden_num), channel_num, 3, 1, activation_fn)
            )
            self.encoder.add_module(
                'conv_{}_{}'.format(idx + 1, 1), conv_block(channel_num, channel_num, 3, 1, activation_fn)
            )
            if idx < repeat_num - 1:
                self.encoder.add_module(
                    'down_{}'.format(idx + 1), conv_block(channel_num, channel_num, 3, 2, activation_fn)
                )
        self.decoder = nn.ModuleList()
        for idx in range(repeat_num):
            channel_num = 2 * channel_num + noise_dim if idx == 0 else hidden_num + hidden_num * (repeat_num - idx)
            self.decoder.add_module(
                'conv_{}_{}'.format(idx, 0), conv_block(channel_num, hidden_num, 3, 1, activation_fn)
            )
            self.decoder.add_module(
                'conv_{}_{}'.format(idx, 1), conv_block(hidden_num, hidden_num, 3, 1, activation_fn)
            )
            if idx < repeat_num - 1:
                self.decoder.add_module(
                    'up_{}'.format(idx), nn.Upsample(scale_factor=2, mode='nearest')
                )
                
        self.head = conv_block(hidden_num, input_channel, 3, 1, activation_fn=None)
    
    def forward(self, x):
        self.encoder_list = []
        # Encode
        x = self.encoder.conv_0_0(x)
        for idx in range(self.repeat_num):
            x = self.encoder.__getattr__('conv_{}_{}'.format(idx + 1, 0))(x)
            x = self.encoder.__getattr__('conv_{}_{}'.format(idx + 1, 1))(x)
            self.encoder_list.append(x)
            if idx < self.repeat_num - 1:
                x = self.encoder.__getattr__('down_{}'.format(idx + 1))(x)
        
        # Random noise
        if self.noise_dim > 0:
            random = torch.tensor(
                np.random.uniform(-1, 1, size=(x.shape[0], self.noise_dim, self.min_fea_map_H, self.min_fea_map_H)),
                dtype=torch.float32,
                device=self.device
            )
            x = torch.cat([x, random], dim=1)
        
        # Decode
        for idx in range(self.repeat_num):
            x = torch.cat([x, self.encoder_list[self.repeat_num-1-idx]], dim=1)
            x = self.decoder.__getattr__('conv_{}_{}'.format(idx, 0))(x)
            x = self.decoder.__getattr__('conv_{}_{}'.format(idx, 1))(x)
            if idx < self.repeat_num - 1:
                x = self.decoder.__getattr__('up_{}'.format(idx))(x)
                
        out = self.head(x)
        return out
    
def init_uniform(layer, stdev=0.02):
    layer.load_state_dict(
        {'weight' : torch.tensor(
            np.random.uniform(-stdev * np.sqrt(3), +stdev * np.sqrt(3), layer.state_dict()['weight'].shape),
            dtype=torch.float32
        ), 
         'bias' : torch.tensor(np.zeros(layer.state_dict()['bias'].shape), dtype=torch.float32)
        }
    )

class DCGANDiscriminator(nn.Module):
    
    def __init__(self, input_dim=3, dim=64, activation=nn.functional.leaky_relu, use_bn=True, *args, **kwargs):
        
        super(DCGANDiscriminator, self).__init__(*args, **kwargs)
        self.dim = dim
        
        self.conv_0 = conv_block(input_dim, dim, 5, 2, activation)
        init_uniform(self.conv_0.conv)

        self.conv_1 = conv_block(dim, 2 * dim, 5, 2, activation, use_bn=use_bn)
        init_uniform(self.conv_1.conv)

        self.conv_2 = conv_block(2*dim, 4*dim, 5, 2, activation, use_bn=use_bn)
        init_uniform(self.conv_2.conv)

        self.conv_3 = conv_block(4*dim, 8*dim, 5, 2, activation, use_bn=use_bn)
        init_uniform(self.conv_3.conv)

        self.conv_4 = conv_block(8*dim, 8*dim, 5, 2, activation, use_bn=use_bn)
        init_uniform(self.conv_4.conv)

        self.head = nn.Linear(8*8*8*dim, 1)
        init_uniform(self.head)
    
    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = torch.reshape(x, (x.shape[0], 8*8*8*self.dim))
        out = self.head(x)
        return out
    
def gan_loss(disc_real, disc_fake):
        gen_cost = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(
            disc_fake,
            torch.ones_like(disc_fake, dtype=torch.float32)
        ))
        disc_cost =  0.5 * torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(
            disc_fake,
            torch.zeros_like(disc_fake, dtype=torch.float32)
        )) + 0.5 * torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(
            disc_real,
            torch.ones_like(disc_real, dtype=torch.float32)
        ))
        return gen_cost, disc_cost
    
def compute_loss_2(source_true, target_true, target_g2, discriminator, LAMBDA):
    inp_true = torch.cat([target_true, source_true], dim=0) # true im
    inp_fake = torch.cat([target_g2, source_true], dim=0) # fake pair
    logits_true = discriminator(inp_true)
    logits_fake = discriminator(inp_fake)
    g2_loss_adv, d_loss = wgan_loss(logits_true, logits_fake)
    l1_loss = torch.mean(torch.abs(target_g2 - target_true))
    return g2_loss_adv, l1_loss, d_loss + LAMBDA * calc_gradient_penalty(discriminator, inp_true, inp_fake)

def compute_loss_1(target_true, target_g1):
    g1_loss = torch.mean(torch.abs(target_g1 - target_true))
    return g1_loss

def wgan_loss(disc_real, disc_fake):
    loss = -torch.sum(disc_fake, dim=1).mean(), \
           -(torch.sum(disc_real, dim=1).mean() - torch.sum(disc_fake, dim=1).mean())
    return loss
    
def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand((real_data.shape[0],1, 1, 1))
    alpha = alpha.expand(real_data.shape)
    alpha = alpha.cuda()

    interpolates = alpha * real_data.data + (1.0 - alpha) * fake_data.data

    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty