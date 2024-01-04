import jax
from jax import numpy as jnp
from flax import linen as nn

from rtnf.render import render_path_dm
from rtnf.network import Grid2, posenc
from rtnf.helpers import makedir

from typing import Any, Callable
import functools

import os
import shutil

# Network
class MLP_act_dm(nn.Module):
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    do_skip: bool = True
    ior_den: int = 400
    deg_point: int = 4
  
    @nn.compact
    def __call__(self, x):
        #MLP with ior activation function builtin
        #Also with posenc builtin
        """A simple Multi-Layer Preceptron (MLP) network

        Parameters
        ----------
        x: jnp.ndarray(float32), 
            [batch_size * n_samples, feature], points.
        net_depth: int, 
            the depth of the first part of MLP.
        net_width: int, 
            the width of the first part of MLP.
        activation: function, 
            the activation function used in the MLP.
        out_channel: 
            int, the number of alpha_channels.
        do_skip: boolean, 
            whether or not to use a skip connection

        Returns
        -------
        out: jnp.ndarray(float32), 
            [batch_size * n_samples, out_channel].
        """
        dense_layer = functools.partial(
            nn.Dense, kernel_init=jax.nn.initializers.he_uniform())

        if self.do_skip:
            skip_layer = self.net_depth // 2

        x = posenc(x, self.deg_point)
        inputs = x
        for i in range(self.net_depth):
            x = dense_layer(self.net_width)(x)
            x = self.activation(x)
            if self.do_skip:
                if i % skip_layer == 0 and i > 0:
                    x = jnp.concatenate([x, inputs], axis=-1)
        out = dense_layer(self.out_channel)(x)
        out = self.ior_act(out)

        return out
    
    def ior_act(self, net_output): 
        eta = nn.softplus(jnp.squeeze(net_output)-2)/self.ior_den + 1.
        return eta

# Image plane stuff
def render_img(traced, img):
    img_grid = Grid2(grid_vals = img, cval=0.)
    return img_grid.interp4(traced)
render_imgs = jax.vmap(render_img, in_axes=(1,0))

def trace_to_plane_3d(ray_trace, zval):
    return jnp.stack([jnp.interp(zval, ray_trace[2,:], ray_trace[0,:]), jnp.interp(zval, ray_trace[2,:], ray_trace[1,:])])

ttp = jax.vmap(trace_to_plane_3d, in_axes=(None,0))
trace_to_plane = jax.vmap(ttp, in_axes=(0, None))

# Forward model
def fwd_model_dm(rays, predict_eta, s_vals, imgs_plane, plane_locs): 
    ray_trace2 = render_path_dm(rays, predict_eta, s_vals).reshape(-1, 6, 25)
    traced2 = trace_to_plane(ray_trace2, plane_locs)
    renders2 = render_imgs(traced2, imgs_plane)
    # print('renders shape: ', renders2.shape)
    return ray_trace2, renders2.sum(0)

# Helpers for file management

def load_from_dir_dm(save_dir): 
    npzfile = jnp.load(save_dir + 'fwd_out.npz')
    X = npzfile['true_X']
    eta_true = npzfile['true_eta']
    rays = npzfile['rays']
    s_vals = npzfile['s_vals']
    plane_locs = npzfile['plane_locs']
    imgs_plane = npzfile['imgs_plane']
    target_imgs = npzfile['target_imgs']
    return X, eta_true, rays, s_vals, plane_locs, imgs_plane, target_imgs