import numpy as np

def strategy_loss(x):
    """
    Seems to compute a loss of -10 for selecting the final action, and 0 otherwise

    Goal is to concentrate all probability on the final action
    """
    x = np.exp(x)
    x = x / np.sum(x, axis=-1)
    return -10 * x[-1]

def entropy_loss(x):
    """
    This treats each dimension of x as the logit of a separate bernoulli distribution
    """
    x = 1.0 / (1.0 + np.exp(-x))
    entropy = x * np.log(x) + (1.0 - x) * np.log(1.0 - x)
    return np.sum(entropy, axis=-1)

def sigmoid_loss(x):
    """
    Simply the sum of the logistic function over x
    """
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    return np.sum(0.1 * x**2 - sigmoid, axis=-1)

def tanh_loss(x):
    """
    Same thing, but with tanh activations instead of logistic ones
    """
    return np.sum(x**2 + np.tanh(x), axis=-1)

def l2_loss(x):
    """
    L2 loss, pretty self-explanatory
    """
    return np.sum(x**2, axis=-1)


def l1_loss(x):
    """
    L1 loss, pretty self-explanatory
    """
    return np.sum(np.abs(x), axis=-1)


def optimize(loss):
    pass


if __name__ == '__main__':
    lr = 0.001  # Fixed step sizes (should test schedules)
    p = 0.01  # Fixed perutrbation sizes (should also be on a schedule)
    std = 2  # Variance of initial distribution
    iterations = 10000
    skip = 500  # This just controls the reporting interval

    # loss_fn = strategy_loss
    # loss_fn = entropy_loss
    # loss_fn = sigmoid_loss
    # loss_fn = tanh_loss
    # loss_fn = l2_loss
    loss_fn = l1_loss

    shape = (5,)

    rng = np.random.default_rng()

    # Initialize solution
    if std > 0:
        theta = rng.normal(scale=std, size=shape)
    else:
        theta = np.zeros(shape)

    print(f"0 - {theta}")

    baseline = 0  # Are we always using a baseline?
    for iteration in range(iterations):
        perturbation = rng.integers(0, 2, size=shape)  # SPSA perturbations always have L1 norm of d
        perturbation = 2 * perturbation - 1
        
        loss = loss_fn(theta + p * perturbation)  # The perturbation scale p seems to be fixed here
        delta = lr * perturbation * ((loss - baseline) / p)
        # delta = lr * perturbation * (loss / p)  # Why do we calculate delta separately (also, \delta in the SPSA notation is the perturbation)

        theta = theta - delta
        baseline = loss  # Ideally this would be a moving average, we're not using it at the moment (lack of baseline causes numerical instability)

        if (iteration + 1) % skip == 0:
            print(f"{iteration + 1} - {theta}, loss: {loss}")
