import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

def renderfn(rays, predict_eta, predict_lum, s_vals):
	'''
	Render an image measurement from initial conditions, eta and luminance fields
	Combines Eikonal Ray-tracing with integration of known 3D luminance field predict_lum
	Input: 
	rays: Batched ray initial conditions. See rtnf.truefield.init_rays for how these are generated.
	predict_eta: Function taking 3D coordinates as input and outputting scalar eta value for each coordinate
	predict_lum: Function taking 3D coordinates as input and outputting scalar luminance/emission value for each coordinate. 
	The function predict_lum is integrated over the ray-traced path. 
	s_vals: path length s-values at which to evaluate the ray-traced path integral. 

	Output: 
	ray_trace: Ray-traced ray paths
	ray_lum: Integrated luminance values at the end of the path
    '''
    
	pred_eta_grad_fn = jax.vmap(jax.value_and_grad(predict_eta), in_axes=(0,))

	def ode_fn_lum(y, s):
		ray_pos = y[..., :3]
		ray_dir = y[..., 3:6]

		eta, eta_grad = pred_eta_grad_fn(jnp.reshape(ray_pos, [-1, 3]))
		lum = predict_lum(jnp.reshape(ray_pos, [-1, 3]))
		d_ds_ray_pos = ray_dir / eta[:, None]
		d_ds_ray_dir = eta_grad
		d_ds_ray_lum = lum[:, None]
		out = jnp.concatenate([d_ds_ray_pos, d_ds_ray_dir, d_ds_ray_lum], axis=-1)
		return out

	ray_pos = rays[..., :3]
	ray_dir = rays[..., 3:6]

	n_init = predict_eta(ray_pos)
	ray_dir = ray_dir * n_init[0] / jnp.linalg.norm(ray_dir, ord=2, axis=-1, keepdims=True)

	y_0 = jnp.concatenate([ray_pos, ray_dir, rays[:, 2*3:]], axis=-1)

	y = odeint(ode_fn_lum, y_0, s_vals)

	ray_trace = jnp.moveaxis(y[..., :3], 0, -1)
	# ray_trace = y[..., :6]
	ray_lum = y[-1, :, 6:]

	return ray_trace, ray_lum

def renderfn_rad(rays, predict_eta, predict_lum, s_vals):
	'''
	Render an image measurement from initial conditions, eta and luminance fields. Uses RRTE to adjust the integrand (radiance) according to the index of refraction eta. 
	See renderfn for input/output annotations. 
    '''
    
	pred_eta_grad_fn = jax.vmap(jax.value_and_grad(predict_eta), in_axes=(0,))

	def ode_fn_lum(y, s):
		ray_pos = y[..., :3]
		ray_dir = y[..., 3:3*2]

		eta, eta_grad = pred_eta_grad_fn(jnp.reshape(ray_pos, [-1, 3]))
		lum = predict_lum(jnp.reshape(ray_pos, [-1, 3]))
		d_ds_ray_pos = ray_dir / eta[:, None]
		d_ds_ray_dir = eta_grad
		d_ds_ray_lum = lum[:, None] / (eta[:, None]**2)
		out = jnp.concatenate([d_ds_ray_pos, d_ds_ray_dir, d_ds_ray_lum], axis=-1)
		return out

	ray_pos = rays[..., :3]
	ray_dir = rays[..., 3:2*3]

	n_init = predict_eta(ray_pos)
	ray_dir = ray_dir * n_init[0] / jnp.linalg.norm(ray_dir, ord=2, axis=-1, keepdims=True)

	y_0 = jnp.concatenate([ray_pos, ray_dir, rays[:, 2*3:]], axis=-1)

	y = odeint(ode_fn_lum, y_0, s_vals)

	ray_trace = jnp.moveaxis(y[..., :3], 0, -1)
	ray_lum = y[-1, :, 2*3:]

	return ray_trace, ray_lum

def render_path(rays, predict_eta, s_vals): 
	'''
	Ray trace a path. 
	Input: 
	rays: batch_size * 6 array of ray initial conditions (3 position coordinates followed by 3 direction coordinates)
	predict_eta: function that maps from 3D coordinates to an eta value at that coordinate
	s_vals: s-values to evaluate the ray path at
	'''

	dd = 3

	pred_eta_grad_fn = jax.vmap(jax.value_and_grad(predict_eta), in_axes=(0,))

	def ode_fn_lum(y, s):
		# Integrate ray path as well as accumulated light
		ray_pos = y[..., :dd]
		ray_dir = y[..., dd:dd*2]

		eta, eta_grad = pred_eta_grad_fn(jnp.reshape(ray_pos, [-1, dd]))
		d_ds_ray_pos = ray_dir / eta[:, None]
		d_ds_ray_dir = eta_grad
        
		out = jnp.concatenate([d_ds_ray_pos, d_ds_ray_dir], axis=-1)
		return out

	ray_pos = rays[..., :dd]
	ray_dir = rays[:, dd:2*dd]
	# Initialize rays with correct starting direction magnitude (n); this allows specification of direction vector to be any magnitude
	n_init = predict_eta(ray_pos)
	ray_dir = ray_dir * n_init[0] / jnp.linalg.norm(ray_dir, ord=2, axis=-1, keepdims=True)

	y_0 = jnp.concatenate([ray_pos, ray_dir], axis=-1)

	y = odeint(ode_fn_lum, y_0, s_vals)
	#y = odeint(ode_fn_lum, y_0, s_vals, rtol=1e-10, atol=1e-10)

	# ray_trace = jnp.moveaxis(y, 0, -1)

	return y

def render_path_dm(rays, predict_eta, s_vals): 
	'''
	Ray trace a path. 
	Input: 
	rays: batch_size * 6 array of ray initial conditions (3 position coordinates followed by 3 direction coordinates)
	predict_eta: function that maps from 3D coordinates to an eta value at that coordinate
	s_vals: s-values to evaluate the ray path at
	'''

	dd = 3

	pred_eta_grad_fn = jax.vmap(jax.value_and_grad(predict_eta), in_axes=(0,))

	def ode_fn_lum(y, s):
		# Integrate ray path as well as accumulated light
		ray_pos = y[..., :dd]
		ray_dir = y[..., dd:dd*2]

		eta, eta_grad = pred_eta_grad_fn(jnp.reshape(ray_pos, [-1, dd]))
		d_ds_ray_pos = ray_dir / eta[:, None]
		d_ds_ray_dir = eta_grad
        
		out = jnp.concatenate([d_ds_ray_pos, d_ds_ray_dir], axis=-1)
		return out

	ray_pos = rays[..., :dd]
	ray_dir = rays[:, dd:2*dd]
	# Initialize rays with correct starting direction magnitude (n); this allows specification of direction vector to be any magnitude
	n_init = predict_eta(ray_pos)
	ray_dir = ray_dir * n_init[0] / jnp.linalg.norm(ray_dir, ord=2, axis=-1, keepdims=True)

	y_0 = jnp.concatenate([ray_pos, ray_dir], axis=-1)

	y = odeint(ode_fn_lum, y_0, s_vals, rtol=1e-10, atol=1e-10)

	ray_trace = jnp.moveaxis(y, 0, -1)

	return ray_trace