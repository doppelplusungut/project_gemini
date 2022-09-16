import torch
from torch import nn
from easydict import EasyDict
import numpy as np

from model.cnn import Classifier


class Deconv(nn.Module):

    def __init__(self, CONFIG):
        super().__init__()

        self.conv_model = Classifier(CONFIG)
        # self.conv_model.load_state_dict(torch.load(path))
        conv_model_layer_names = np.array(
            [type(module).__name__ for module in self.conv_model.modules() 
                if not isinstance(module, nn.Sequential) and not isinstance(module, Classifier)]
        )

        pool_idx = np.where(conv_model_layer_names == 'MaxPool2d')[0]
        conv_idx = np.where(conv_model_layer_names == 'Conv2d')[0]

        # Initialize all deconv_blocks
        self.deconv_blocks = []

        for i, idx in enumerate(list(pool_idx)):
            self.conv_model.model[idx].return_indices = True

            # for each maxpool layer create a deconv block with self._build_deconv
            pool_cfg = {
                'kernel_size': self.conv_model.model[idx].kernel_size,
                'stride': self.conv_model.model[idx].stride,
                'padding': self.conv_model.model[idx].padding,
            }
            conv_cfg = {
                'out_channels': self.conv_model.model[conv_idx[i]].in_channels,
                'in_channels': self.conv_model.model[conv_idx[i]].out_channels,
                'kernel_size': self.conv_model.model[conv_idx[i]].kernel_size,
                'stride': self.conv_model.model[conv_idx[i]].stride,
                'padding': self.conv_model.model[conv_idx[i]].padding,
            }

            self.deconv_blocks.append(self._build_deconv(pool_cfg, conv_cfg))

            # for each maxpool layer create hook to extract feature output vector
            # (maybe not needed if we enumerate through self.conv_model layers in forward!)

    def forward(self, x, n):
        conv_weights = []
        conv_bias = []
        pool_idx = []
        pool_input_size = []
        conv_input_size = []
        block_output = []
        i = 0
        # forward pass of self.conv_model: get outputs of each conv block and indices from each max pool layer
        for module in self.conv_model.modules():
            if type(module).__name__ == 'Conv2d':
                conv_input_size.append(x.shape)
                x = module(x)
                conv_weights.append(module.weight)
                conv_bias.append(module.bias)
            
            elif type(module).__name__ == 'MaxPool2d':
                pool_input_size.append(x.shape)
                x, idx = module(x)
                pool_idx.append(idx)
                block_output.append(x)
            
            elif isinstance(module, nn.Sequential) or isinstance(module, Classifier):
                pass

            else:
                x = module(x) 

        # set input for the first deconv block to the output of the n-th block in the forward network
        x = block_output[n-1]

        # remove the last len(list)-n entries from each list and reserve them
        self.deconv_blocks = self.deconv_blocks[:n]
        conv_weights = conv_weights[:n]
        conv_bias = conv_bias[:n]
        pool_idx = pool_idx[:n]
        pool_input_size = pool_input_size[:n]
        conv_input_size = conv_input_size[:n]

        self.deconv_blocks.reverse()
        conv_weights.reverse()
        conv_bias.reverse()
        pool_idx.reverse()
        pool_input_size.reverse()
        conv_input_size.reverse()

        # for n-th conv block: use its output as input for the len(pool_idx)-n'th deconv block, and use indices from forward pass of self.conv_model for unpooling
        for i, block in enumerate(self.deconv_blocks):
            x = block['unpool'](x, pool_idx[i], output_size=pool_input_size[i])
            x = block['relu'](x)
            
            # 'transpose' weights and biases of the conv layers in the original model
            with torch.no_grad():
                #TODO: have to zero out filters, not sure which ones yet!
                block['conv'].weight.copy_(conv_weights[i])#.flip([2,3])) # TODO: figure out if we need to flip the filters or not
                # block['conv'].bias = conv_bias[i]
                block['conv'].bias.copy_(torch.zeros(conv_weights[i].shape[1]))
            x = block['conv'](x, output_size=conv_input_size[i])

        # return the output of the last deconv block
        return x



    def _build_deconv(self, pool_cfg, conv_cfg):
        model = {
            'unpool': nn.MaxUnpool2d(**pool_cfg),
            'relu': nn.ReLU(),
            #'conv': nn.Conv2d(**conv_cfg)
            'conv': nn.ConvTranspose2d(**conv_cfg)
        }
        return model
