# directly from Patrick Kidger
# at (https://github.com/patrick-kidger/diffrax/blob/main/examples/latent_ode.ipynb)
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
import optax

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
        dt0 = 0.5  # selected as a reasonable choice for this problem
        y0 = self.latent_to_hidden(latent)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return jax.vmap(self.hidden_to_data)(sol.ys)

    # output hidden variables
    def hidden_sample(self, ts, latent):
        dt0 = 0.5  # selected as a reasonable choice for this problem
        y0 = self.latent_to_hidden(latent)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    @staticmethod
    def _loss(ys, pred_ys, mean, std):
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # KL(N(mean, std^2) || N(0, 1))
        variational_loss = 0.5 * jnp.sum(mean**2 + std**2 - 2 * jnp.log(std) - 1)
        # late time penalty
        incriment = jnp.linspace(0, 0.25, len(ys))
        time_loss = 0# incriment * jnp.sum((ys - pred_ys) ** 2)
        """
        Idea, why don't we put more weight on the later time predictions?
        """
        return reconstruction_loss + variational_loss + time_loss

    # Run both encoder and decoder during training.
    def train(self, ts, ys, *, key):
        latent, mean, std = self._latent(ts, ys, key)
        pred_ys = self._sample(ts, latent)
        return self._loss(ys, pred_ys, mean, std)

    # Run just the decoder during inference.
    def sample(self, ts, *, key):
        latent = jr.normal(key, (self.latent_size,))
        return self._sample(ts, latent)

def get_data(dataset_size, *, key):
    ykey, tkey1, tkey2 = jr.split(key, 3)

    #y0 = jr.normal(ykey, (dataset_size, 2))
    y0 = jr.uniform(ykey, (dataset_size, 2), minval=1, maxval=1)
    t0 = 0
    #t1 = 2 + jr.uniform(tkey1, (dataset_size,))
    t1 = 10 + 10 * jr.uniform(tkey1, (dataset_size,), minval=0, maxval=1)
    ts = jr.uniform(tkey2, (dataset_size, 20)) * (t1[:, None] - t0) + t0
    ts = jnp.sort(ts)
    dt0 = 0.5

    def func(t, y, args):
        k = 0.5
        return jnp.array([[-k, 1.0], [-1, -k]]) @ y

    def vector_field(t, y, args):
        prey, predator = y
        a, b, c, d = args
        d_prey = a * prey - b * prey * predator
        d_predator = -d * predator + c * prey * predator
        d_y = jnp.array([d_prey, d_predator])
        return d_y
    args = (2/3, 4/3, 1, 1)

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

def func(t, y, args):
        k = 0.5
        return jnp.array([[-k, 1.0], [-1, -k]]) @ y


def vector_field(t, y, args):
    prey, predator = y
    a, b, c, d = args
    d_prey = a * prey - b * prey * predator
    d_predator = -d * predator + c * prey * predator
    d_y = jnp.array([d_prey, d_predator])
    return d_y

def solve(ts, y0):
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        ts[0],
        ts[-1],
        0.1,
        y0,
        args = (2/3, 4/3, 1, 1),
        saveat=diffrax.SaveAt(ts=ts),
    )
    return sol.ys

def main(
    dataset_size=50000,
    batch_size=256,
    lr=1e-2,
    steps=300,
    save_every=100,
    hidden_size=32,
    latent_size=32,
    width_size=32,
    depth=2,
    seed=5678,
):
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key, train_key, sample_key = jr.split(key, 5)

    ts, ys = get_data(dataset_size, key=data_key)

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
    num_plots = 1 + (steps - 1) // save_every
    if ((steps - 1) % save_every) != 0:
        num_plots += 1
    fig, axs = plt.subplots(2, num_plots, figsize=(num_plots * 12, 12))
    idx=0
    axs[0][idx].set_ylabel("x")
    for step, (ts_i, ys_i) in zip(
        range(steps), dataloader((ts, ys), batch_size, key=loader_key)
    ):
        start = time.time()
        value, model, opt_state, train_key = make_step(
            model, opt_state, ts_i, ys_i, train_key
        )
        end = time.time()
        print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")

        if (step % save_every) == 0 or step == steps - 1:
            ax = axs[0][idx]
            # Sample over a longer time interval than we trained on. The model will be
            # sufficiently good that it will correctly extrapolate!
            sample_t = jnp.linspace(0, 60, 300)
            sample_y = model.sample(sample_t, key=sample_key)
            sample_t = np.asarray(sample_t)
            sample_y = np.asarray(sample_y)
            exact_y = solve(sample_t, sample_y[0,:])
            #ax.plot(sample_t, sample_y, color='firebrick')
            #ax.scatter(sample_t, exact_y, color='firebrick', s=5)
            ax.plot(sample_t, sample_y[:, 0], color='firebrick', label='predator')
            ax.plot(sample_t, sample_y[:, 1], color='steelblue', label='prey')
            ax.scatter(sample_t, exact_y[:, 0], color='firebrick', s=3)
            ax.scatter(sample_t, exact_y[:, 1], color='steelblue', s=3)
            #ax.set_xticks([])
            #ax.set_yticks([])
            ax.set_title(f"training step: {step}")
            ax.set_xlabel("t")
            ax.legend()

            # now the phase space plot
            ax = axs[1][idx]
            ax.plot(sample_y[:, 0], sample_y[:, 1], color='gray', label='LatendODE')
            ax.scatter(exact_y[:, 0], exact_y[:, 1], color='gray', s=3, label='exact')
            #ax.set_title(f"training step: {step}")
            ax.set_xlabel("predator")
            ax.set_ylabel("prey")
            ax.legend()
            idx+=1


    plt.suptitle("Latent ODE Model: Lotka-Volterra Equations", fontsize=20)
    plt.savefig("latent_ode.png")
    plt.show()


main()
