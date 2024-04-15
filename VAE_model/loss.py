import optax


def loss_fn(params, X, model):
    X_hat = model.apply(params, X)
    return 2 * optax.l2_loss(X_hat, X).mean()
