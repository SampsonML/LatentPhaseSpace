# ------------------------------------------ #
# The architecture of the deep autoencoder   #
# ------------------------------------------ #

import numpy as np
import jax.numpy as jnp
import jax
import functools
from flax import linen as nn


class Encoder(nn.Module):
    """
    Inputs
    ------
    bottleneck: int - the number of latent variables

    Outputs
    -------
    x: jnp.array - the latent variables
    """
    bottleneck: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(5)(x)
        x = nn.selu(x)
        x = nn.Dense(features=self.bottleneck)(x)
        return x

class Decoder(nn.Module):
    """
    Inputs
    ------
    out: int - the number of output features

    Outputs
    -------
    x: jnp.array - the reconstructed data
    """
    out: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(5)(x)
        x = nn.selu(x)
        x = nn.Dense(features=self.out)(x)
        return x


class AutoEncoder(nn.Module):
    """
    Inputs
    ------
    bottleneck: int - the number of latent variables
    out: int - the number of output features

    Outputs
    -------
    x_hat: jnp.array - the reconstructed data
    """
    bottleneck: int
    out: int
    def setup(self):
        # Alternative to @nn.compact -> explicitly define modules
        # Better for later when we want to access the encoder and decoder explicitly
        self.encoder = Encoder(bottleneck=self.bottleneck)
        self.decoder = Decoder(out=self.out)

    def __call__(self, x):

        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat