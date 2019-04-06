import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import conv,upconv

class DCGenerator(nn.Module):
    def __init__(self, noise_size, conv_dim):
        super(DCGenerator, self).__init__()

        self.conv_dim = conv_dim
        self.linear_bn = upconv(noise_size, conv_dim * 4, 5, stride=4)
        self.upconv1 = upconv(conv_dim * 4, conv_dim * 2, 5)
        self.upconv2 = upconv(conv_dim * 2, conv_dim, 5)
        self.upconv3 = upconv(conv_dim, 3, 5, batch_norm=False)

    def forward(self, z):
        """Generates an image given a sample of random noise.
            Input
            -----
                z: BS x noise_size x 1 x 1   -->  BSx100x1x1 (during training)
            Output
            ------
                out: BS x channels x image_width x image_height  -->  BSx3x32x32 (during training)
        """
        batch_size = z.size(0)

        out = F.relu(self.linear_bn(z))  # .view(-1, self.conv_dim * 4, 4, 4)  # BS x 128 x 4 x 4
        out = F.relu(self.upconv1(out))  # BS x 64 x 8 x 8
        out = F.relu(self.upconv2(out))  # BS x 32 x 16 x 16
        out = F.tanh(self.upconv3(out))  # BS x 3 x 32 x 32

        out_size = out.size()
        if out_size != torch.Size([batch_size, 3, 32, 32]):
            raise ValueError("expect {} x 3 x 32 x 32, but get {}".format(batch_size, out_size))
        return out

class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64):
        super(DCDiscriminator, self).__init__()
        self.conv1 = conv(3, conv_dim, 5)
        self.conv2 = conv(conv_dim, conv_dim * 2, 5)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 5)
        self.conv4 = conv(conv_dim * 4, 1, 5, padding=1, batch_norm=False)

    def forward(self, x):
        batch_size = x.size(0)

        out = F.relu(self.conv1(x))  # BS x 64 x 16 x 16
        out = F.relu(self.conv2(out))  # BS x 64 x 8 x 8
        out = F.relu(self.conv3(out))  # BS x 64 x 4 x 4

        out = self.conv4(out).squeeze()
        out_size = out.size()
        if out_size != torch.Size([batch_size, ]):
            raise ValueError("expect {} x 1, but get {}".format(batch_size, out_size))
        return out