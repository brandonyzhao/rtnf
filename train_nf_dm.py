from flax import linen as nn
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint

import jax
import jax.numpy as jnp

import optax

from tqdm import tqdm

import os
from datetime import datetime
import pickle
import pdb

import matplotlib.pyplot as plt

from rtnf.network import *
from rtnf.helpers import init_savedir
from rtnf.dark_matter import load_from_dir_dm, MLP_act_dm
from rtnf.optimization import get_X_bd, lossfn
from rtnf.dark_matter import fwd_model_dm
from rtnf.optimization import train_step, pred_eta_step
from rtnf.plots_dm import make_slice_plots

import pdb
import argparse

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=str, default='dark_matter')
parser.add_argument('-e', type=str, default='debug')
parser.add_argument('-d', type=str, default='0')
                  
args = parser.parse_args()

from jax.config import config
config.update("jax_enable_x64", True)

base_dir, save_dir, exp_run = init_savedir(args.b, args.e)

os.environ['CUDA_VISIBLE_DEVICES'] = args.d
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# seed for MLP initialization
seed = 0
rand_key = jax.random.PRNGKey(seed)

# Iteration schedule
num_iters = 50000
loss_freq = 100
plot_freq = 100
save_freq = 100

# Hyperparameters
batch_size = 4096
lam = 1.
bd_val = 1.

write_dir = './runs/' + exp_run + '.{}'.format(datetime.now().strftime('%Y-%m-%d.%H:%M:%S'))
writer = SummaryWriter(write_dir)

X, eta_true, rays, s_vals, plane_locs, imgs_plane, target_imgs = load_from_dir_dm(save_dir)
X_reg = get_X_bd(16)
target_img = target_imgs.sum(0)

eta_MLP = MLP_act_dm(net_depth=4,
              net_width=256,
              activation=nn.elu,
              out_channel=1,
              do_skip=False, 
              ior_den = 4e5, 
              deg_point = 4)

lr_init = 1e-4
lr_final = 5e-6
tx = optax.adam(lambda x : jnp.exp(jnp.log(lr_init) * (1 - (x/num_iters)) + jnp.log(lr_final) * (x/num_iters)))

# Initialize parameters
mlp_params_eta = eta_MLP.init(rand_key, jnp.ones([3]))['params']
opt_state = train_state.TrainState.create(apply_fn=eta_MLP.apply, params=mlp_params_eta, tx=tx)

lossfn = jax.tree_util.Partial(lossfn, lam=lam, bd_val=bd_val)

render_eta = jax.tree_util.Partial(fwd_model_dm, s_vals=s_vals, imgs_plane=imgs_plane, plane_locs=plane_locs)

make_slice_plots(eta_true, writer, 0, 'True Eta Slice Plot')

for i in (pbar := tqdm(range(1, num_iters+1))):
  if i==1 or i % loss_freq == 0:
    eta = pred_eta_step(X, opt_state).reshape(eta_true.shape)
    eta_diff_norm = jnp.linalg.norm(eta - eta_true)
    writer.add_scalar('Eta Diff', eta_diff_norm, i)

  if (i==1 or i % plot_freq == 0):
    make_slice_plots(eta, writer, i, 'Model Eta Slice Plot')
    
  # Batchify things
  batch_ind = jax.random.choice(rand_key, rays.shape[0], shape=(batch_size,), replace=False)
  rays_batch = jnp.take(rays, batch_ind, 0)
  target_img_batch = jnp.take(target_img.flatten(), batch_ind, 0)
  # Training Step
  loss, _, opt_state, ray_trace, ray_lum, rand_key, grad = train_step(rays_batch, target_img_batch, X_reg, opt_state, render_eta, lossfn, rand_key)
  log_loss = jnp.log10(loss)
  writer.add_scalar('Log Loss', log_loss, i)

  if i % save_freq == 0: 
      save_checkpoint(save_dir, opt_state, i, prefix='model_MLP', keep=1, overwrite=True)
      pickle.dump(eta, open(save_dir+'eta_out/eta_out_%04d.p'%(i//save_freq), 'wb'))