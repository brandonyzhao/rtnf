import flax
from flax import linen as nn
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint

import jax
import jax.numpy as jnp

import optax

import os
import time
from datetime import datetime
import pickle
from tqdm import tqdm

from rtnf.network import *
from rtnf.truefield import interp_grid
from rtnf.render import renderfn
from rtnf.helpers import load_from_dir_np, shard_things, init_savedir
from rtnf.optimization import train_pstep, lossfn, test_pstep_and_plot
from rtnf.plots import make_slice_plots

import argparse

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=str, default='debug')
parser.add_argument('-e', type=str, default='debug')
parser.add_argument('-d', type=str, default='0')
args = parser.parse_args()

base_dir, save_dir, exp_run = init_savedir(args.b, args.e)
write_dir = './runs/' + exp_run + '.{}'.format(datetime.now().strftime('%Y-%m-%d.%H:%M:%S'))
os.environ['CUDA_VISIBLE_DEVICES'] = args.d
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# seed for MLP initialization
seed = 0
# Training parameters
num_iters = 50000
test_freq = 100
save_freq = 100

# Tensorboard writer
writer = SummaryWriter(write_dir, write_to_disk=True)

# Load data from true generated field
X_test, rays, s_vals, lum_field, ray_lum_target, eta_true = load_from_dir_np(save_dir)
# Interpolate emission on a grid
predict_lum = interp_grid(lum_field, cval=0.)

# from rtnf_old.helpers import load_from_dir
# from rtnf_old.truefield import fit_ls_to_grid

# X_test, rays, s_vals, ls, ray_lum_target, eta_true = load_from_dir(save_dir, cache=False)
# predict_lum = fit_ls_to_grid(ls, 64)

# Boundary points for regularization
from rtnf.optimization import get_X_bd
X_reg = get_X_bd(16)

ray_lum_target_plot = ray_lum_target.sum(1) 
rand_key = jax.random.PRNGKey(seed)


# Store emission grid as state in rendering function
render_eta = jax.tree_util.Partial(renderfn, predict_lum=predict_lum, s_vals=s_vals)

# Instantiate MLP
eta_MLP = MLP_gas(net_depth=4,
                  net_width=256,
                  activation=nn.elu,
                  out_channel=1,
                  do_skip=False,
                  deg_point=6)

# Initialize parameters
mlp_params_eta = eta_MLP.init(rand_key, jnp.ones([3]))['params']

# Create optimizer
lr_init = 1e-4
lr_final = 5e-6
tx = optax.adam(lambda x : jnp.exp(jnp.log(lr_init) * (1 - (x/num_iters)) + jnp.log(lr_final) * (x/num_iters)))
opt_state = train_state.TrainState.create(apply_fn=eta_MLP.apply, params=mlp_params_eta, tx=tx)

# Set loss function & weights
from rtnf.optimization import lossfn
lam = 0.1
bd_val = 1.0003
lossfn = jax.tree_util.Partial(lossfn, lam=lam, bd_val=bd_val)

# Parallelization
opt_state = flax.jax_utils.replicate(opt_state)
rand_key = jax.random.split(rand_key, jax.local_device_count())
rays_shard, ray_lum_target_shard, X_reg_shard, X_test_shard = shard_things(rays, ray_lum_target, X_reg, X_test)

# Plot ground truth eta field
make_slice_plots(eta_true, writer, 0, 'True Eta')

diff_vmax = None
start = time.time()

for i in tqdm(range(1, num_iters+1)):
  if i==1 or i % test_freq == 0:
    # Test step with recording & plots
    start, diff_vmax, eta = test_pstep_and_plot(i, start, writer, diff_vmax, ray_lum_target_plot, rays_shard, ray_lum_target_shard, X_reg_shard, X_test_shard, opt_state, render_eta, eta_true, rand_key, lossfn, plot_img=True)

  if (i % save_freq == 0) or (i == 1): 
    # save_checkpoint(save_dir + 'model_out/', opt_state, i, prefix='model_MLP', keep=1, overwrite=True)
    pickle.dump(eta, open(save_dir+'eta_out/eta_out_%04d.p'%(i), 'wb'))

  # Training
  loss_train, loss2, opt_state, ray_trace, rendering, rand_key, grad = train_pstep(rays_shard, ray_lum_target_shard, X_reg_shard, opt_state, render_eta, lossfn, rand_key)