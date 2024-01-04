import jax
import jax.numpy as jnp
import jax.random as random

import numpy as np

import time
from functools import partial

from rtnf.plots import plots_regular, plots_eta

def get_X_bd(res): 
  # Get boundary points for regularization
  # Evenly spaced points on the faces of a cube
  lsp = jnp.linspace(0, 1, res)
  x, y, z = jnp.meshgrid(lsp, lsp, lsp, indexing='xy')
  pts = jnp.stack([x, y, z], axis=-1).reshape(-1, 3)
  on_bd0 = jnp.any(pts==0, axis=1)
  on_bd1 = jnp.any(pts==1, axis=1)
  X_reg = pts[on_bd0 | on_bd1]
  return X_reg

def lossfn(apply_fn, params, render_eta, rays, ray_lum_target, X, lam, bd_val):
  predict_eta = lambda x: apply_fn({'params': params}, x)
  ray_trace, ray_lum = render_eta(rays, predict_eta)

  loss = jnp.mean(jnp.square(ray_lum - ray_lum_target))
  eta = predict_eta(X)
  reg_term = jnp.sum(jnp.square(eta - bd_val)) * lam
  loss_reg = loss + reg_term

  return loss_reg, [ray_trace, ray_lum, loss]

@partial(jax.jit, static_argnums=(4,5))
def train_step(rays, ray_lum_target, x, opt_state, render_eta, lossfn, key):
  key, new_key = random.split(key)
  apply_fn = opt_state.apply_fn
  params = opt_state.params
  vals, grad = jax.value_and_grad(lossfn, argnums=(1), has_aux=True)(apply_fn, params, render_eta, rays, ray_lum_target, x)
  #grad = jax.lax.pmean(grad, axis_name='batch')
  loss, [ray_trace, rendering, loss2] = vals
  opt_state = opt_state.apply_gradients(grads=grad)

  return loss, loss2, opt_state, ray_trace, rendering, new_key, grad

train_pstep = jax.pmap(train_step, axis_name='batch', in_axes=(0,0,0,0,None,None,0), static_broadcasted_argnums=(4,5))

@partial(jax.jit, static_argnums=(4,5))
def test_step(rays, ray_lum_target, x, opt_state, render_eta, lossfn, key):
  key, new_key = random.split(key)
  apply_fn = opt_state.apply_fn
  params = opt_state.params
  loss, [ray_trace, ray_lum, loss2] = lossfn(apply_fn, params, render_eta, rays, ray_lum_target, x)
  # do not update opt_state
  return loss, loss2, opt_state, ray_trace, ray_lum, new_key

test_pstep = jax.pmap(test_step, axis_name='batch', in_axes=(0,0,0,0,None,None,0), static_broadcasted_argnums=(4,5))

@jax.jit
def pred_eta_step(x, opt_state): 
  predict_eta = lambda x: opt_state.apply_fn({'params': opt_state.params}, x)
  eta_out = predict_eta(x)
  return eta_out

pred_eta_pstep = jax.pmap(pred_eta_step, axis_name='batch', in_axes=0)

def test_pstep_and_plot(i, start, writer, diff_vmax, ray_lum_target_plot, rays_shard, ray_lum_target_shard, X_reg_shard, X_test_shard, opt_state, render_eta, eta_true, rand_key, lossfn, plot_img=True): 
  res = 64
  
  loss, loss2, opt_state, ray_trace, ray_lum, rand_key = test_pstep(rays_shard, ray_lum_target_shard, X_reg_shard, opt_state, render_eta, lossfn, rand_key)
  eta = pred_eta_pstep(X_test_shard, opt_state)

  ray_lum_plot = ray_lum.flatten()

  eta_max = eta.max()
  loss = np.mean(loss)
  loss2 = np.mean(loss2)
  log_loss = np.log(loss)
  
  eta = eta.reshape(eta_true.shape)
  eta_diff_norm = jnp.linalg.norm(eta - eta_true)

  if plot_img: 
    plots_regular(res, ray_lum_plot, ray_lum_target_plot, eta, eta_true, writer, i)
  else:
    plots_eta(res, ray_lum_plot, ray_lum_target_plot, eta, eta_true, writer, i)
  start = time.time()

  writer.add_scalar('Log Loss', log_loss, i)
  writer.add_scalar('Eta Diff', eta_diff_norm, i)
  writer.add_scalar('MSE Loss', loss2, i)
  writer.add_scalar('Reg Term', loss - loss2, i)

  return start, diff_vmax, eta