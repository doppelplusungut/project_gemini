# import torch
from urllib.parse import parse_qsl
from torch import nn
import numpy as np

from cnn import Classifier


class Deconv(nn.Module):

    def __init__(self, path=None, n_out=61):
        super().__init__()

        conv_model = Classifier(n_out=n_out)
        # conv_model.load_state_dict(torch.load(path))
        conv_model_layer_names = np.array(
            [type(module).__name__ for module in conv_model.modules() 
                if not isinstance(module, nn.Sequential) and not isinstance(module, Classifier)]
        )

        pool_idx = np.where(conv_model_layer_names == 'MaxPool2d')[0]

        for i, idx in enumerate(list(pool_idx)):
            conv_model.model[idx].return_indices = True

            # for each maxpool layer create a deconv block with self._build_deconv

            # for each maxpool layer create hook to extract feature output vector
            # (maybe not needed if we enumerate through conv_model layers in forward!)


        print("bye")

    def forward(self, x, n):
        # forward pass of conv_model: get outputs of each conv block and indices from each max pool layer

        # for n-th conv block: use its output as input for the len(pool_idx)-n'th deconv block, and use indices from forward pass of conv_model for unpooling

        # return the output of the last deconv block

        pass


    def _build_deconv(self):
        pass

Deconv()