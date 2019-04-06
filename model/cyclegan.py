import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import conv,upconv,ResnetBlock

class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture.
    """
    def __init__(self, conv_dim=64, init_zero_weights=False):
        super(CycleGenerator, self).__init__()
        # Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = conv(3, conv_dim, 5, init_zero_weights=init_zero_weights)
        self.conv2 = conv(conv_dim, conv_dim * 2, 5)
        # Define the transformation part of the generator
        self.resnet_block = ResnetBlock(conv_dim * 2)
        # Define the decoder part of the generator (that builds up the output image from features)
        self.upconv1 = upconv(conv_dim * 2, conv_dim, 5)
        self.upconv2 = upconv(conv_dim, 3, 5, batch_norm=False)

    def forward(self, x):
        """Generates an image conditioned on an input image.
            Input
            -----
                x: BS x 3 x 32 x 32
            Output
            ------
                out: BS x 3 x 32 x 32
        """
        batch_size = x.size(0)
        out = F.relu(self.conv1(x))  # BS x 32 x 16 x 16
        out = F.relu(self.conv2(out))  # BS x 64 x 8 x 8
        out = F.relu(self.resnet_block(out))  # BS x 64 x 8 x 8
        out = F.relu(self.upconv1(out))  # BS x 32 x 16 x 16
        out = F.tanh(self.upconv2(out))  # BS x 3 x 32 x 32
        out_size = out.size()
        if out_size != torch.Size([batch_size, 3, 32, 32]):
            raise ValueError("expect {} x 3 x 32 x 32, but get {}".format(batch_size, out_size))
        return out