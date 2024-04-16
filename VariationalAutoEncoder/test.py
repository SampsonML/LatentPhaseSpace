# work from https://nipunbatra.github.io/blog/posts/autoencoder.html
# for dataloaders see https://www.kaggle.com/code/aakashnain/building-models-in-jax-part2-flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
import jax.random as random
from flax import linen as nn
import pandas as pd

from AE import AutoEncoder, Encoder, Decoder
from train import train_step

# Matt's plotting params
# ---------------------------------------------- #
import matplotlib as mpl
mpl.rcParams["xtick.top"] = True
mpl.rcParams["ytick.right"] = True
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["xtick.major.size"] = 7
mpl.rcParams["xtick.minor.size"] = 4.5
mpl.rcParams["ytick.major.size"] = 7
mpl.rcParams["ytick.minor.size"] = 4.5
mpl.rcParams["xtick.major.width"] = 2
mpl.rcParams["xtick.minor.width"] = 1.5
mpl.rcParams["ytick.major.width"] = 2
mpl.rcParams["ytick.minor.width"] = 1.5
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams.update({"text.usetex": True})
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
bn = 6  # bottleneck size: the amount of latent variables
out_size = X.shape[1]

# instantiate the autoencoder
ae = AutoEncoder(bn, out_size)

# initialise parameters
params = ae.init(random.PRNGKey(0), X)
X_hat = ae.apply(params, X)

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


# run the model training
optimized_params, loss_array = train_step(
    X,
    optax.adam(learning_rate=0.05),
    ae,
    jax.random.PRNGKey(0),
    n_iter=500,
    print_every=20,
)

# plot the loss
plt.plot(loss_array)
plt.xlabel("Iterations")
_ = plt.ylabel("Reconstruction loss")
plt.show()

# plot the result
plot_2d_reconstruction(X, optimized_params, ae, True)
