# Matt Sampson, Latent Space Least Action Models
# Testing grounds
# initial latentODE architecture modified from Patrick Kidger/Ricky Chen
import time
import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy._typing import _32Bit
from numpy.lib.shape_base import row_stack
import optax
import os

# turn on float 64
# from jax import config
# config.update("jax_enable_x64", True)

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


class Func(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __call__(self, t, y, args):
        return self.scale * self.mlp(y)


class LatentODE(eqx.Module):
    func: Func
    rnn_cell: eqx.nn.GRUCell

    hidden_to_latent: eqx.nn.Linear
    latent_to_hidden: eqx.nn.MLP
    hidden_to_data: eqx.nn.Linear

    hidden_size: int
    latent_size: int

    def __init__(
        self, *, data_size, hidden_size, latent_size, width_size, depth, key, **kwargs
    ):
        super().__init__(**kwargs)

        mkey, gkey, hlkey, lhkey, hdkey = jr.split(key, 5)

        scale = jnp.ones(())
        mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=mkey,
        )
        self.func = Func(scale, mlp)
        self.rnn_cell = eqx.nn.GRUCell(data_size + 1, hidden_size, key=gkey)

        self.hidden_to_latent = eqx.nn.Linear(hidden_size, 2 * latent_size, key=hlkey)
        self.latent_to_hidden = eqx.nn.MLP(
            latent_size, hidden_size, width_size=width_size, depth=depth, key=lhkey
        )
        self.hidden_to_data = eqx.nn.Linear(hidden_size, data_size, key=hdkey)

        self.hidden_size = hidden_size
        self.latent_size = latent_size

    # Encoder of the VAE
    def _latent(self, ts, ys, key):
        data = jnp.concatenate([ts[:, None], ys], axis=1)
        hidden = jnp.zeros((self.hidden_size,))
        for data_i in reversed(data):
            hidden = self.rnn_cell(data_i, hidden)
        context = self.hidden_to_latent(hidden)
        mean, logstd = context[: self.latent_size], context[self.latent_size :]
        std = jnp.exp(logstd)
        latent = mean + jr.normal(key, (self.latent_size,)) * std
        return latent, mean, std

    # Decoder of the VAE
    def _sample(self, ts, latent):
        dt0 = 0.125  # selected as a reasonable choice for this problem
        y0 = self.latent_to_hidden(latent)
        solver = (
            diffrax.Bosh3()
        )  # see: https://docs.kidger.site/diffrax/api/solvers/ode_solvers/
        adjoint = (
            diffrax.RecursiveCheckpointAdjoint()
        )  # see: https://docs.kidger.site/diffrax/api/adjoints/
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            solver,
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
            adjoint=adjoint,
        )
        return jax.vmap(self.hidden_to_data)(sol.ys)

    @staticmethod
    def _loss(ys, pred_ys, mean, std):
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # KL(N(mean, std^2) || N(0, 1))
        variational_loss = 0.5 * jnp.sum(mean**2 + std**2 - 2 * jnp.log(std) - 1)
        return reconstruction_loss + variational_loss

    @staticmethod
    def _latentloss(self, ts, ys, pred_ys, pred_latent, mean, std, key):
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # KL(N(mean, std^2) || N(0, 1))
        variational_loss = 0.5 * jnp.sum(mean**2 + std**2 - 2 * jnp.log(std) - 1)
        # Malhanobis distance between latents \sqrt{(x - y)^T \Sigma^{-1} (x - y)}
        diff = jnp.diff(pred_latent, axis=0)
        Cov = jnp.eye(diff.shape[1]) * std # for now identity
        Cov = jnp.linalg.inv(Cov)
        d_latent = jnp.sqrt(jnp.sum( jnp.dot(diff , Cov) * diff, axis=1))
        d_latent = jnp.sum(d_latent)
        #jax.debug.print("size of latent dist is {}", d_latent)
        #jax.debug.print("latent distance: {}", d_latent)
        alpha = 1 # weighting parameter for distance penalty
        return reconstruction_loss + variational_loss + alpha * d_latent

    # Run both encoder and decoder during training.
    def train(self, ts, ys, *, key):
        latent, mean, std = self._latent(ts, ys, key)
        pred_ys = self._sample(ts, latent)
        pred_latent = self._sampleLatent(ts, latent)
    #    return self._loss(ys, pred_ys, mean, std)
        return self._latentloss(self, ts, ys, pred_ys, pred_latent, mean, std, key)


    # Run just the decoder during inference.
    def sample(self, ts, *, key):
        latent = jr.normal(key, (self.latent_size,))
        return self._sample(ts, latent)

    def _sampleLatent(self, ts, latent):
        dt0 = 0.1  # selected as a reasonable choice for this problem
        y0 = self.latent_to_hidden(latent)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Bosh3(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return jax.vmap(self.hidden_to_latent)(sol.ys)

    def sampleLatent(self, ts, *, key):
        latent = jr.normal(key, (self.latent_size,))
        return self._sampleLatent(ts, latent)


def get_data(dataset_size, *, key, func=None, t_end=1, n_points=100):
    ykey, tkey1, tkey2 = jr.split(key, 3)
    y0 = jr.uniform(ykey, (dataset_size, 2), minval=0, maxval=5)
    t0 = 0
    t1 = t_end + 1 * jr.uniform(tkey1, (dataset_size,), minval=0, maxval=1)
    ts = jr.uniform(tkey2, (dataset_size, n_points)) * (t1[:, None] - t0) + t0
    ts = jnp.sort(ts)
    dt0 = 0.1

    # ------------------------
    # Lotka-Volterra equations
    def LVE(t, y, args):
        prey, predator = y
        a, b, c, d = args
        d_prey = a * prey - b * prey * predator
        d_predator = -d * predator + c * prey * predator
        d_y = jnp.array([d_prey, d_predator])
        return d_y

    LVE_args = (1, 0.5, 1, 0.50)  # a, b, c, d

    # --------------------------
    # Simple harmonic oscillator
    def SHO(t, y, args):
        y1, y2 = y
        theta = args
        dy1 = y2
        dy2 = -y1 - theta * y2
        d_y = jnp.array([dy1, dy2])
        return d_y

    SHO_args = (0.15)  # theta

    # --------------------------------------
    # Periodically forced hamonic oscillator
    def PFHO(t, y, args):
        y1, y2 = y
        w, b, k, force = args
        dy1 = y2
        dy2 = force * jnp.cos(w * t) - b * y2 - k * y1
        d_y = jnp.array([dy1, dy2])
        return d_y

    PFHO_args = (1, 1, 1, 3)  # w, b, k, force

    if func == "LVE":
        vector_field = LVE
        args = LVE_args
    elif func == "SHO":
        vector_field = SHO
        args = SHO_args
    elif func == "PFHO":
        vector_field = PFHO
        args = PFHO_args
    else:
        raise ValueError("func must be one of 'LVE', 'SHO', 'PFHO'")

    def solve(ts, y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            args=args,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    ys = jax.vmap(solve)(ts, y0)

    return ts, ys


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while start < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def main(
    dataset_size=20000,
    batch_size=256,
    n_points=100,
    lr=1e-2,
    steps=30,
    plot_every=10,
    save_every=10,
    hidden_size=8,
    latent_size=2,
    width_size=8,
    depth=2,
    seed=1992,
    func="PFHO",
    figname="latent_ODE.png",
):
    # Defining vector fields again for use in comparison testing
    # ------------------------
    # Lotka-Volterra equations
    def LVE(t, y, args):
        prey, predator = y
        a, b, c, d = args
        d_prey = a * prey - b * prey * predator
        d_predator = -d * predator + c * prey * predator
        d_y = jnp.array([d_prey, d_predator])
        return d_y

    LVE_args = (1, 0.5, 1, 0.5)  # a=prey-growth, b, c, d

    # --------------------------
    # Simple harmonic oscillator
    def SHO(t, y, args):
        y1, y2 = y
        theta = args
        dy1 = y2
        dy2 = -y1 - theta * y2
        d_y = jnp.array([dy1, dy2])
        return d_y

    SHO_args = 0.15  # theta

    # --------------------------------------
    # Periodically forced hamonic oscillator
    def PFHO(t, y, args):
        y1, y2 = y
        w, b, k, force = args
        dy1 = y2
        dy2 = force * jnp.cos(w * t) - b * y2 - k * y1
        d_y = jnp.array([dy1, dy2])
        return d_y

    PFHO_args = (1, 1, 1 , 3)  # w, b, k, force

    if func == "LVE":
        vector_field = LVE
        args = LVE_args
        rows = 3
        TITLE = "Latent ODE Model: Lotka-Volterra Equations"
        LAB_X = "Prey"
        LAB_Y = "Predator"
    elif func == "SHO":
        vector_field = SHO
        args = SHO_args
        rows = 4
        TITLE = "Latent ODE Model: Simple Harmonic Oscillator"
        LAB_X = "Position"
        LAB_Y = "Velocity"
    elif func == "PFHO":
        vector_field = PFHO
        args = PFHO_args
        rows = 3
        TITLE = "Latent ODE Model: Periodically Forced Harmonic Oscillator"
        LAB_X = "Position"
        LAB_Y = "Velocity"
    else:
        raise ValueError("func must be one of 'LVE', 'SHO', 'PFHO'")

    def solve(ts, y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            0.1,
            y0,
            args=args,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key, train_key, sample_key = jr.split(key, 5)

    # get the data
    ts, ys = get_data(dataset_size, key=data_key, func=func, t_end=20, n_points=n_points)

    model = LatentODE(
        data_size=ys.shape[-1],
        hidden_size=hidden_size,
        latent_size=latent_size,
        width_size=width_size,
        depth=depth,
        key=model_key,
    )

    @eqx.filter_value_and_grad
    def loss(model, ts_i, ys_i, key_i):
        batch_size, _ = ts_i.shape
        key_i = jr.split(key_i, batch_size)
        loss = jax.vmap(model.train)(ts_i, ys_i, key=key_i)
        return jnp.mean(loss)

    @eqx.filter_jit
    def make_step(model, opt_state, ts_i, ys_i, key_i):
        value, grads = loss(model, ts_i, ys_i, key_i)
        key_i = jr.split(key_i, 1)[0]
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return value, model, opt_state, key_i

    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # Plot results
    num_plots = 1 + (steps - 1) // plot_every
    if ((steps - 1) % plot_every) != 0:
        num_plots += 1
    fig, axs = plt.subplots(rows, num_plots, figsize=(num_plots * 4, rows * 4 - 2))
    idx = 0
    for step, (ts_i, ys_i) in zip(
        range(steps), dataloader((ts, ys), batch_size, key=loader_key)
    ):
        start = time.time()
        value, model, opt_state, train_key = make_step(
            model, opt_state, ts_i, ys_i, train_key
        )
        end = time.time()
        print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")

        # save parameters
        SAVE_DIR = "saved_models"
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        if (step % save_every) == 0 or step == steps - 1:
            fn = SAVE_DIR + "/latentODE" + str(step) + ".eqx"
            eqx.tree_serialise_leaves(fn, model)

        if (step % plot_every) == 0 or step == steps - 1:
            # create some sample trajectories
            t_end = 40
            ext = 20
            sample_t = jnp.linspace(0, t_end, 300)
            sample_y = model.sample(sample_t, key=sample_key)
            sample_latent = model.sampleLatent(sample_t, key=sample_key)
            sample_latent = np.asarray(sample_latent)
            sample_t = np.asarray(sample_t)
            sample_y = np.asarray(sample_y)
            exact_y = solve(sample_t, sample_y[0, :])
            sz = 2
            # plot the trajectories in data space
            ax = axs[0][idx]
            ax.plot(sample_t, sample_y[:, 0], color="firebrick", label=LAB_X)
            ax.plot(sample_t, sample_y[:, 1], color="steelblue", label=LAB_Y)
            ax.scatter(sample_t, exact_y[:, 0], color="firebrick", s=sz)
            ax.scatter(sample_t, exact_y[:, 1], color="steelblue", s=sz)
            ax.set_title(f"training step: {step}")
            ax.set_xlabel("t")
            ax.axvspan(ext, t_end + 2, alpha=0.2, color="coral")
            ax.set_xlim([0, t_end])
            if idx == 0:
                ax.set_ylabel("arb")
                ax.legend()

            # the phase space plot
            ax = axs[1][idx]
            sample_y_in = sample_y[sample_t < ext]
            sample_y_out = sample_y[sample_t >= ext]
            exact_y_in = exact_y[sample_t < ext]
            exact_y_out = exact_y[sample_t >= ext]
            ax.plot(
                sample_y_in[:, 0],
                sample_y_in[:, 1],
                color="darkgray",
                label="LatentODE",
            )
            ax.scatter(
                exact_y_in[:, 0],
                exact_y_in[:, 1],
                color="darkgray",
                s=sz,
                label="exact",
            )
            ax.plot(
                sample_y_out[:, 0],
                sample_y_out[:, 1],
                color="coral",
                label="ODE: extrapolated",
            )
            ax.scatter(exact_y_out[:, 0], exact_y_out[:, 1], color="coral", s=sz)
            ax.set_xlabel(LAB_X)
            if idx == 0:
                ax.set_ylabel(LAB_Y)
                ax.legend()

            # now the latent space plot
            ax = axs[2][idx]
            cmap = plt.get_cmap("plasma")
            for i in range(sample_latent.shape[1]):
                name = f"latent{i}"
                color = cmap(i / sample_latent.shape[1])
                ax.plot(sample_t, sample_latent[:, i], color=color, label=name)
            ax.set_xlabel("time")
            ax.axvspan(ext, t_end + 2, alpha=0.2, color="coral")
            ax.set_xlim([0, t_end])
            if idx == 0:
                ax.set_ylabel("arb")
                ax.legend()

            if rows > 3:
                ax = axs[3][idx]
                latent_in = sample_latent[sample_t < ext]
                latent_out = sample_latent[sample_t >= ext]
                ax.plot(
                    latent_in[:, 0],
                    latent_in[:, 1],
                    color="darkgray",
                    label="LatentODE",
                )
                ax.plot(
                    latent_out[:, 0],
                    latent_out[:, 1],
                    color="coral",
                    label="ODE: extrapolated",
                )
                ax.set_xlabel("latent 0")
                if idx == 0:
                    ax.set_ylabel("latent 1")
            idx += 1

    plt.suptitle(TITLE, y=0.935, fontsize=20)
    plt.savefig(figname, bbox_inches="tight", dpi=200)


# run the code
main(
    n_points=50,
    lr=1e-2,
    steps=300,
    plot_every=100,
    save_every=100,
    hidden_size=4,
    latent_size=1,
    width_size=4,
    func="SHO",
    figname="latentPlot.png",
)
