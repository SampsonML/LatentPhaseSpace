import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from loss import loss_fn


# training function
def train_step(
    X: jnp.array,
    optimizer: optax._src.base.GradientTransformation,
    model: nn.Module,
    key_param: jax.random.PRNGKey,
    n_iter: int = 500,
    print_every: int = 10,
):
    loss_array = np.zeros(n_iter)
    params = model.init(key_param, X)
    opt_state = optimizer.init(params)
    loss_grad_fn = jax.value_and_grad(loss_fn)

    for i in range(n_iter):
        loss_val, grads = loss_grad_fn(params, X, model)
        loss_array[i] = loss_val.item()
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if i % print_every == 0:
            print("Loss step {}: ".format(i), loss_val)
    return params, loss_array
