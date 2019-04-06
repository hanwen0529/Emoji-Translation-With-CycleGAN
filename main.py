from utils.data_utils import get_file
import numpy as np
import torch
from model.train_utils import AttrDict, print_opts
from model.train import train

data_fpath = get_file(fname='emojis',
                      origin='http://www.cs.toronto.edu/~jba/emojis.tar.gz',
                      untar=True)

print("File path is:",data_fpath)  # default file path is 'data/emojis'

SEED = 11
# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
args = AttrDict()

'''
# Train Vanilla GAN
args_dict = {
              'image_size':32,
              'g_conv_dim':32,
              'd_conv_dim':64,
              'noise_size':100,
              'num_workers': 0,
              'train_iters':1,
              'X':'Windows',  # options: 'Windows' / 'Apple'
              'Y': None,
              'lr':0.0003,
              'beta1':0.5,
              'beta2':0.999,
              'batch_size':32,
              'checkpoint_dir': 'checkpoints_gan',
              'sample_dir': 'samples_gan',
              'load': None,
              'log_step':200,
              'sample_every':200,
              'checkpoint_every':1000,
}
args.update(args_dict)

print_opts(args)
G, D = train(args)
'''

# Train CycleGAN
args_dict = {
              'image_size':32,
              'g_conv_dim':32,
              'd_conv_dim':32,
              'init_zero_weights': False,
              'num_workers': 0,
              'train_iters':5000,
              'X':'Apple',
              'Y':'Windows',
              'lambda_cycle': 0.015,
              'lr':0.0003,
              'beta1':0.3,
              'beta2':0.999,
              'batch_size':32,
              'checkpoint_dir': 'checkpoints_cyclegan',
              'sample_dir': 'samples_cyclegan',
              'load': None,
              'log_step':200,
              'sample_every':200,
              'checkpoint_every':1000,
}
args.update(args_dict)

print_opts(args)
G_XtoY, G_YtoX, D_X, D_Y = train(args)
