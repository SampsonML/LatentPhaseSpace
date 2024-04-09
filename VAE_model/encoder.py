# ----------------------------------- #
# The encoder module of the VAE model.#
# ----------------------------------- #

import numpy as np
import jax.numpy as jnp
import jax
import functools

def random_vec(size):
    return np.random.normal(size, scale=1)

def encoder(x, theta):
    """The encoder takes as input x and gives out probability of z,
    expressed as normal distribution parameters. Assuming each z dim is independent,
    output |z| x 2 matrix"""
    w1, w2, w3, b1, b2, b3 = theta
    hx = jax.nn.relu(w1 @ x + b1)
    hx = jax.nn.relu(w2 @ hx + b2)
    out = w3 @ hx + b3
    # slice out stddeviation and make it positive
    reshaped = out.reshape((-1, 2))
    # we slice with ':' to keep rank same
    std = jax.nn.softplus(reshaped[:, 1:])
    mu = reshaped[:, 0:1]
    return jnp.concatenate((mu, std), axis=1)


def init_theta(input_dim, hidden_units, latent_dim):
    """Create inital theta parameters"""
    w1 = random_vec(size=(hidden_units, input_dim))
    b1 = np.zeros(hidden_units)
    w2 = random_vec(size=(hidden_units, hidden_units))
    b2 = np.zeros(hidden_units)
    # need to params per dim (mean, std)
    w3 = random_vec(size=(latent_dim * 2, hidden_units))
    b3 = np.zeros(latent_dim * 2)
    return [w1, w2, w3, b1, b2, b3]

# to test for now
latent_dim = 1
hidden_dim = 16
input_dim = 50
data_test = np.random.normal(size=(input_dim))
# test them
theta = init_theta(input_dim, hidden_dim, latent_dim)
print(theta)
#encoder(data_test, theta)



