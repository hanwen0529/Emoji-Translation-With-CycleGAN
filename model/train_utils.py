import os
import torch
from dcgan import DCGenerator,DCDiscriminator
from cyclegan import CycleGenerator
from utils.data_utils import to_var

def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)

def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    """Prints model information for the generators and discriminators.
    """
    if G_YtoX:
        print("                 G_XtoY                ")
        print("---------------------------------------")
        print(G_XtoY)
        print("---------------------------------------")

        print("                 G_YtoX                ")
        print("---------------------------------------")
        print(G_YtoX)
        print("---------------------------------------")

        print("                  D_X                  ")
        print("---------------------------------------")
        print(D_X)
        print("---------------------------------------")

        print("                  D_Y                  ")
        print("---------------------------------------")
        print(D_Y)
        print("---------------------------------------")
    else:
        print("                 G                     ")
        print("---------------------------------------")
        print(G_XtoY)
        print("---------------------------------------")

        print("                  D                    ")
        print("---------------------------------------")
        print(D_X)
        print("---------------------------------------")

def create_model(opts):
    """Builds the generators and discriminators.
    """
    if opts.Y is None:
        ### GAN
        G = DCGenerator(noise_size=opts.noise_size, conv_dim=opts.g_conv_dim)
        D = DCDiscriminator(conv_dim=opts.d_conv_dim)

        print_models(G, None, D, None)

        if torch.cuda.is_available():
            G.cuda()
            D.cuda()
            print('Models moved to GPU.')
        return G, D

    else:
        ### CycleGAN
        G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
        G_YtoX = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
        D_X = DCDiscriminator(conv_dim=opts.d_conv_dim)
        D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)

        print_models(G_XtoY, G_YtoX, D_X, D_Y)

        if torch.cuda.is_available():
            G_XtoY.cuda()
            G_YtoX.cuda()
            D_X.cuda()
            D_Y.cuda()
            print('Models moved to GPU.')
        return G_XtoY, G_YtoX, D_X, D_Y

def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def gan_checkpoint(iteration, G, D, opts):
    """Saves the parameters of the generator G and discriminator D.
    """
    G_path = os.path.join(opts.checkpoint_dir, 'G_{}.pkl'.format(iteration))
    D_path = os.path.join(opts.checkpoint_dir, 'D_{}.pkl'.format(iteration))
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)

def cyclegan_checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts):
    """Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_X, D_Y.
    """
    G_XtoY_path = os.path.join(opts.checkpoint_dir, 'G_XtoY_{}.pkl'.format(iteration))
    G_YtoX_path = os.path.join(opts.checkpoint_dir, 'G_YtoX_{}.pkl'.format(iteration))
    D_X_path = os.path.join(opts.checkpoint_dir, 'D_X_{}.pkl'.format(iteration))
    D_Y_path = os.path.join(opts.checkpoint_dir, 'D_Y_{}.pkl'.format(iteration))
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)


def load_checkpoint(opts, iteration):
    """Loads the generator and discriminator models from checkpoints.
    """
    G_XtoY_path = os.path.join(opts.load, 'G_XtoY_{}.pkl'.format(iteration))
    G_YtoX_path = os.path.join(opts.load, 'G_YtoX_{}.pkl'.format(iteration))
    D_X_path = os.path.join(opts.load, 'D_X_{}.pkl'.format(iteration))
    D_Y_path = os.path.join(opts.load, 'D_Y_{}.pkl'.format(iteration))

    G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    G_YtoX = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    D_X = DCDiscriminator(conv_dim=opts.d_conv_dim)
    D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)

    G_XtoY.load_state_dict(torch.load(G_XtoY_path, map_location=lambda storage, loc: storage))
    G_YtoX.load_state_dict(torch.load(G_YtoX_path, map_location=lambda storage, loc: storage))
    D_X.load_state_dict(torch.load(D_X_path, map_location=lambda storage, loc: storage))
    D_Y.load_state_dict(torch.load(D_Y_path, map_location=lambda storage, loc: storage))

    if torch.cuda.is_available():
        G_XtoY.cuda()
        G_YtoX.cuda()
        D_X.cuda()
        D_Y.cuda()
        print('Models moved to GPU.')
    return G_XtoY, G_YtoX, D_X, D_Y