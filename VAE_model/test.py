# work from https://nipunbatra.github.io/blog/posts/autoencoder.html
import jax
import jax.numpy as jnp
import numpy as np
import optax

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import jax.random as random
from flax import linen as nn
import pandas as pd

from AE import AutoEncoder, Encoder, Decoder

# Matt's plotting params
# ---------------------------------------------- #
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.major.size"] = 7
plt.rcParams["xtick.minor.size"] = 4.5
plt.rcParams["ytick.major.size"] = 7
plt.rcParams["ytick.minor.size"] = 4.5
plt.rcParams["xtick.major.width"] = 2
plt.rcParams["xtick.minor.width"] = 1.5
plt.rcParams["ytick.major.width"] = 2
plt.rcParams["ytick.minor.width"] = 1.5
plt.rcParams["axes.linewidth"] = 2
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams.update({"text.usetex": True})
# ---------------------------------------------- #

# load up a trial of the spring data
path = "/Users/mattsampson/Research/PrincetonThesis/LatentDynamicalModel/hamiltonianGenerator/spring_data/"
trial = "trial_k2_m2.npy"
data = np.load(path + trial, allow_pickle=True)
data = data.T
plt.scatter(data[:, 0], data[:, 1])
plt.title("Spring data")
plt.show()

# create a copy of the data
X = data.copy()
bn = 10  # bottleneck size: the amount of latent variables

enc = Encoder(bottleneck=bn)
dec = Decoder(out=2)

params_enc = enc.init(random.PRNGKey(0), X)
X_bottlenecked = enc.apply(params_enc, X)

out_size = X.shape[1]
ae = AutoEncoder(bn, out_size)


# initialise parameters
params = ae.init(random.PRNGKey(0), X)

X_hat = ae.apply(params, X)

# Encoded values/latent representation
encoded_1d = Encoder(bn).apply({"params": params["params"]["encoder"]}, X).flatten()


# define plot function
def plot_2d_reconstruction(X, params, model, trained=False):
    X_hat = model.apply(params, X)
    plt.scatter(X[:, 0], X[:, 1], label="Original Data")
    plt.scatter(X_hat[:, 0], X_hat[:, 1], label="Reconstructed Data")
    if trained:
        plt.title("Trained")
    else:
        plt.title("Untrained")
    plt.show()


# plot the result
plot_2d_reconstruction(X, params, ae, trained=False)


# define loss function
def loss(params, X_hat):
    X_hat = ae.apply(params, X)
    return 2 * optax.l2_loss(X, X_hat).mean()


print(loss(params, X_hat))


# training function
def train(
    X: jnp.array,
    optimizer: optax._src.base.GradientTransformation,
    model: nn.Module,
    key_param: jax.random.PRNGKey,
    n_iter: int = 500,
    print_every: int = 10,
):
    loss_array = np.zeros(n_iter)

    def loss(params, X):
        X_hat = model.apply(params, X)
        #return 2 * optax.l2_loss(X_hat, X).mean()
        return optax.squared_error(X_hat, X).mean()

    params = model.init(key_param, X)
    opt_state = optimizer.init(params)
    loss_grad_fn = jax.value_and_grad(loss)

    for i in range(n_iter):
        loss_val, grads = loss_grad_fn(params, X)
        loss_array[i] = loss_val.item()
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if i % print_every == 0:
            print("Loss step {}: ".format(i), loss_val)
    return params, loss_array


optimized_params, loss_array = train(
    X,
    optax.adam(learning_rate=0.05),
    ae,
    jax.random.PRNGKey(0),
    n_iter=500,
    print_every=20,
)

plt.plot(loss_array)
plt.xlabel("Iterations")
_ = plt.ylabel("Reconstruction loss")
plt.show()

plot_2d_reconstruction(X, optimized_params, ae, True)
