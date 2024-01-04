import jax
import os
import shutil

from jax import numpy as jnp

def shard(xs):
    """Split data into shards for multiple devices along the first dimension."""
    return jax.tree_util.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)

def shard_things(*args): 
    return (shard(arg) for arg in args)

def makedir(path): 
	if not os.path.exists(path):
		os.makedirs(path)                

def init_savedir(b, e): 
    base_dir = './exp/' + b + '/'
    assert os.path.exists(base_dir), "base dir doesn't exist!"
    exp_run = b + '_' + e
    save_dir = './exp/' + exp_run + '/'
    if not os.path.exists(save_dir): 
        print('New Experiment run, copying base dir to new save dir')
        shutil.copytree(base_dir, save_dir)
    makedir(save_dir + 'eta_out/')
    return base_dir, save_dir, exp_run

def save_to_dir_np(save_dir, X_test, rays, s_vals, lum_field, ray_lum_target, eta_true):
    jnp.save(save_dir + 'X_test.npy', X_test, allow_pickle=False)
    jnp.save(save_dir + 'rays.npy', rays, allow_pickle=False)
    jnp.save(save_dir + 's_vals.npy', s_vals, allow_pickle=False)
    jnp.save(save_dir + 'lum_field.npy', lum_field, allow_pickle=False)
    jnp.save(save_dir + 'ray_lum_target.npy', ray_lum_target, allow_pickle=False)
    jnp.save(save_dir + 'eta_true.npy', eta_true, allow_pickle=False)

def load_from_dir_np(base_dir):
    X_test = jnp.load(base_dir + 'X_test.npy', allow_pickle=False)
    rays = jnp.load(base_dir + 'rays.npy', allow_pickle=False)
    s_vals = jnp.load(base_dir + 's_vals.npy', allow_pickle=False)
    lum_field = jnp.load(base_dir + 'lum_field.npy', allow_pickle=False)
    ray_lum_target = jnp.load(base_dir + 'ray_lum_target.npy', allow_pickle=False)
    eta_true = jnp.load(base_dir + 'eta_true.npy', allow_pickle=False)

    return X_test, rays, s_vals, lum_field, ray_lum_target, eta_true