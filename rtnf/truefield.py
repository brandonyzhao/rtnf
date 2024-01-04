import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from scipy.stats import special_ortho_group, qmc

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objects as go

from rtnf.network import Grid2, Light_SourcesCov

def rotate_cov(cov, size, seed):
    og = special_ortho_group(dim=3, seed=seed)
    evecs = og.rvs(size=size)
    return jnp.swapaxes(evecs, 1, 2) @ (cov @ evecs)

def init_rays(fov_angle, res, num_aux=1): 
	fov_angle = fov_angle * jnp.pi / 180
	angle_y = 0.5 - jnp.tan(fov_angle / 2)
	num_rays = res**2
	lsx, lsy = jnp.meshgrid(jnp.linspace(0.5-angle_y, 0.5+angle_y, res),
					jnp.linspace(0.5-angle_y, 0.5+angle_y, res), indexing='xy')
	ap_pts = jnp.stack([lsx, lsy, jnp.zeros([res, res])], axis=-1)
	lsbx, lsby = jnp.meshgrid(jnp.linspace(0., 1, res), jnp.linspace(0., 1, res), indexing='xy')
	rays_b = jnp.stack([lsbx, lsby, jnp.ones([res, res])], axis=-1)
	rays_d = rays_b - ap_pts
	ap_pts = ap_pts.reshape([-1, 3])
	rays_d = rays_d.reshape([-1, 3])

	rays_l = jnp.zeros((num_rays, num_aux))

	rays = jnp.concatenate([ap_pts, rays_d, rays_l], axis=-1)

	return rays

def sample_gaussian(seed, X_eta, length_scale=0.25):
    kernel = Matern(length_scale=length_scale, nu=1.5)
    gpc = GaussianProcessRegressor(kernel=kernel, random_state=0)
    matern_out = gpc.sample_y(X_eta, random_state=seed)
    matern_out = jnp.array(matern_out)

    return matern_out

def generate_field(ior_type, seed, X_eta, eta_shape, mag=1):
	def ior_act(net_output, mag=1): 
		eta = mag * nn.softplus(jnp.squeeze(net_output)-5)/50 + 1.
		return eta
	if ior_type == 'matern': 
		matern_out = sample_gaussian(seed, X_eta)
		matern_out = matern_out.reshape(eta_shape)
		grid_out = ior_act(matern_out, mag=mag)
		eta_Grid = Grid2(grid_vals = grid_out, cval=1.)
		predict_eta = eta_Grid.interp5
	else: 
		raise Exception("Invalid IoR type")
	return grid_out, predict_eta

def interp_grid(grid, cval): 
	grid_obj = Grid2(grid_vals = grid, cval=cval)
	predict_interp = lambda x: grid_obj.interp5(x)
	return predict_interp

def field_and_pred(source_locs, covs, X_lum, lum_shape): 
	ls = Light_SourcesCov(source_locs, covs)
	lum_field = ls.predict_lum(X_lum).sum(0).reshape(lum_shape)
	lum_Grid = Grid2(grid_vals = lum_field, cval=0.)
	predict_lum_grid = lambda x: lum_Grid.interp5(x)

	return lum_field, predict_lum_grid, ls

def generate_emission(ls_type, seed, X_lum, lum_shape, **kwargs): 
	# Sample Source Locations
	if ls_type == 'poisson': 
		rand_gen = np.random.default_rng(seed)
		engine = qmc.PoissonDisk(d=3, radius=0.13, seed=rand_gen)
		source_locs = engine.random(1000)
		num_sources = source_locs.shape[0]	
	elif ls_type == 'uniform': 
		num_sources = kwargs['num_sources']
		rand_key = jax.random.PRNGKey(seed)
		sources = []
		while len(sources) < 500: 
			source_loc = random.uniform(rand_key, (1,3), minval=0., maxval=1.) * jnp.array([[1, 1, 1.]])
			rand_key, _ = random.split(rand_key)
			sources.append(jnp.squeeze(source_loc))
		source_locs = jnp.stack(sources[:num_sources], axis=0)
	else: 
		raise Exception("Invalid Light Source type")
	
	# Generate Elliptical Covariances
	cov = jnp.array([[[2, 0, 0], [0., 0.5, 0], [0, 0, 0.5]]]) * 1e-3
	covs = rotate_cov(cov, num_sources, seed)

	return field_and_pred(source_locs, covs, X_lum, lum_shape)

def plot_3d(volume, res): 
	X, Y, Z = np.mgrid[:res, :res, :res]

	Z = res - 1 - Z
	X = res - 1 - X

	fig = go.Figure(data=go.Volume(
		x=Z.flatten(), y=X.flatten(), z=Y.flatten(),
		value=volume.flatten(),
		opacity=0.05,
		surface_count=10,
		))
	fig.update_layout(scene_xaxis_showticklabels=False,
					scene_yaxis_showticklabels=False,
					scene_zaxis_showticklabels=False)

	fig.show()

def plot_2d(image, title='', save=False, filename=''):
	res = image.shape[0]
	plt.xlim([0, res-1])
	plt.ylim([0, res-1])
	sns.heatmap(image)
	plt.title(title)
	if save:
		plt.savefig(filename)
	else:
		plt.show()
	plt.close()