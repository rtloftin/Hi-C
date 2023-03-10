import numpy as np

def entropy_loss(x):
    x = 1.0 / (1.0 + np.exp(-x))
    entropy = x * np.log(x) + (1.0 - x) * np.log(1.0 - x)
    return np.sum(entropy, axis=-1)

def sigmoid_loss(x):
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    return np.sum(0.1 * x**2 - sigmoid, axis=-1)

def tanh_loss(x):
    return np.sum(x**2 + np.tanh(x), axis=-1)

def l2_loss(x):
    return np.sum(x**2, axis=-1)


def l1_loss(x):
    return np.sum(np.abs(x), axis=-1)


if __name__ == '__main__':
    lr = 0.000001
    p = 0.01
    std = 0.5
    iterations = 100000
    skip = 1000

    loss_fn = entropy_loss
    # loss_fn = sigmoid_loss
    # loss_fn = tanh_loss
    # loss_fn = l2_loss
    # loss_fn = l1_loss

    shape = (5,)

    rng = np.random.default_rng()

    # Initialize solution
    if std > 0:
        theta = rng.normal(scale=std, size=shape)
    else:
        theta = np.zeros(shape)

    print(f"0 - {theta}")

    for iteration in range(iterations):
        perturbation = rng.integers(0, 2, size=shape)
        perturbation = 2 * perturbation - 1
        loss = loss_fn(theta + p * perturbation)
        delta = lr * perturbation * (loss / p)
        theta = theta - delta

        if (iteration + 1) % skip == 0:
            print(f"{iteration + 1} - {theta}. p: {perturbation}, delta: {np.abs(delta)}, loss: {loss}")
