import numpy as np

def l2_loss(x):
    return np.sum(x**2, axis=-1)


def l1_loss(x):
    return np.sum(np.abs(x), axis=-1)


if __name__ == '__main__':
    lr = 0.001
    p = 0.05
    std = 0.5
    iterations = 10000
    skip = 100

    loss_fn = l2_loss
    # loss_fn = l1_loss

    shape = (5,)

    rng = np.random.default_rng(0)

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
        theta = theta - lr * perturbation * (loss / p)

        if (iteration + 1) % skip == 0:
            print(f"{iteration + 1} - {theta}. p: {perturbation}, loss: {loss}")
