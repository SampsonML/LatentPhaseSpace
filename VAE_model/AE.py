# ----------------------------------- #
# The encoder module of the VAE model.#
# ----------------------------------- #

import numpy as np
import jax.numpy as jnp
import jax
import functools
from flax import linen as nn


class Encoder(nn.Module):
    bottleneck: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(5)(x)
        #x = nn.selu(x)
        x = nn.selu(x)
        x = nn.Dense(features=self.bottleneck)(x)
        return x

class Decoder(nn.Module):
    out: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(5)(x)
        x = nn.selu(x)
        x = nn.Dense(features=self.out)(x)
        return x


class AutoEncoder(nn.Module):
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
