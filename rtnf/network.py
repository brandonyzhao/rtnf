import jax
from flax import linen as nn
from jax import numpy as jnp

import numpy as np

from typing import Any, Callable
import functools

safe_sin = lambda x: jnp.sin(x % (100 * jnp.pi))

class Light_SourcesCov():
    """
    Class for evaluating scaled multivariate Gaussian pdf (used for 
    simulating elliptical light sources). The PDF for multiple points 
    is evaluated efficiently in a vectorized way. Much of the code was 
    taken from StackOverflow, although I can't find the original post anymore. 

    Parameters
    ----------
    source_locs: jnp.ndarray, 
        the locations of the light sources
    covs: jnp.ndarray, 
        covariance matrix for each light source

    """
    def __init__(self, source_locs, covs): 
        self.source_locs = source_locs
        self.covs = covs
        # Calculate these ahead of time to save time on forward
        vals, vecs = jnp.linalg.eigh(self.covs)
        valsinvs = 1./vals
        self.Us = vecs * jnp.sqrt(valsinvs)[:, None]
        self.logdets = jnp.sum(jnp.log(vals), axis=1)

    def multiple_logpdfs_vec_input(self, xs):
        devs = xs[:, None, :] - self.source_locs[None, :, :]
        devUs = jnp.einsum('jnk,nki->jni', devs, self.Us)
        mahas = jnp.sum(jnp.square(devUs), axis=2)
        dim    = xs.shape[1]
        log2pi = jnp.log(2 * jnp.pi)
        out = -0.5 * (dim * log2pi + mahas + self.logdets[None, :])
        return out.T

    def predict_lum(self, ray_pos, damp=200.): 
        """
        Evaluates the emission function (sum of all the gaussian pdfs) at locations given by ray_pos. 

        Parameters
        ----------
        ray_pos: jnp.ndarray, 
            array of positions at which to evaluate the emission function
        damp: float, 
            optional scaling factor for the emission function
        """
        logpdf = self.multiple_logpdfs_vec_input(ray_pos)
        return jnp.exp(logpdf) / damp
    
def posenc(x, deg):
    """
    Concatenate `x` with a positional encoding of `x` with degree `deg`.
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Parameters
    ----------
    x: jnp.ndarray, 
        variables to be encoded. Note that x should be in [-pi, pi].
    deg: int, 
        the degree of the encoding.

    Returns
    -------
    encoded: jnp.ndarray, 
        encoded variables.
    """
    if deg == 0:
        return x
    scales = jnp.array([2**i for i in range(deg)])
    xb = jnp.reshape((x[..., None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1])
    four_feat = safe_sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    return jnp.concatenate([x] + [four_feat], axis=-1)
    
class MLP_act(nn.Module):
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    do_skip: bool = True
    ior_den: int = 400
    deg_point: int = 4
  
    @nn.compact
    def __call__(self, x):
        """
        A simple Multi-Layer Perceptron (MLP) network. Also includes
        positional encoding (posenc) as well as a final activation function
        (ior_act) to transform the output into a refractive index.

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
        eta = nn.sigmoid(jnp.squeeze(net_output)) / (0.5 * self.ior_den) + 1.
        return eta
    
class MLP_gas(nn.Module):
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    do_skip: bool = True
    deg_point: int = 4
  
    @nn.compact
    def __call__(self, x):
        """
        A simple Multi-Layer Perceptron (MLP) network. Also includes
        positional encoding (posenc) as well as a final activation function
        (ior_act) to transform the output into a refractive index. The 
        final activation function differs from MLP_act to match the 
        parameters of the gas flow simulation.

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
        eta = 1.0003 - (nn.sigmoid(jnp.squeeze(net_output)) * 3e-4)
        return eta

class Grid2(): 
    """
    A linearly interpolated grid. Used for voxel interpolation (interp5), as well as 
    image plane interpolation (interp4) for dark matter experiments. 

    Parameters
    ----------
    grid_vals: jnp.ndarray
        The voxel points to interpolate between. 
    cval: 
        The constant value to fill past the edges of the voxel grid. 
    """
    def __init__(self, grid_vals, cval=0.): 
        self.grid_vals = grid_vals
        self.grid_res = grid_vals.shape[0]
        self.cval = cval
    def interp4(self, x, order=1): 
        # image-plane interpolation for [0,1] coordinates
        coords = (x * self.grid_res) - 0.5
        coords = jnp.flip(coords, axis=-1)
        out = jax.scipy.ndimage.map_coordinates(self.grid_vals, coords.T, order=order, cval=self.cval)
        return out
    def interp5(self, x, order=1): 
        # Fixing boundary conditions for true eta field
        coords = (x * (self.grid_res+1)) - 1
        #want to swap x and y
        perm = jnp.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
        coords = coords @ perm
        out = jax.scipy.ndimage.map_coordinates(self.grid_vals, coords.T, order=order, cval=self.cval)
        return out

def get_X(res): 
	low = 0.5 / res
	high = 1 - (0.5 / res)
    
	linsp = np.linspace(low, high, res)
	x, y, z = np.meshgrid(linsp, linsp, linsp, indexing='xy')
	pts = np.stack([x, y, z], axis=-1)
	X = pts.reshape((-1, 3))
	return X