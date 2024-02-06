import numpy as np
import torch

from hi_c.learners.schedule import get_schedule


class HiC:
    """Uncoupled hierarchical gradients with simultaneous perturbations + commitments (Loftin et al., AAMAS 2024)"""

    def __init__(self,
                 game,  # Generic learners need access to the game to differentiate through it
                 player_id,  # Tell's the agent which player to differentiate through
                 lr=0.005,  # Learning rate, or learning rate schedule
                 p=0.001,  # Perturbation size schedule
                 k=10,  # Commitment schedule
                 baseline_lambda=1.,
                 burn_in=0,
                 rng=None,
                 device="cpu"):
        self._game = game
        self._player_id = player_id
        self._baseline_lambda = baseline_lambda
        self._burn_in = burn_in
        self._device = device

        # Get parameter space
        self._space = game.strategy_spaces[player_id]

        # Construct random number generator if none provided
        self._rng = rng if rng is not None else np.random.default_rng()

        # Configure learning-rate, perturbation, and commitment schedules
        self._lr = get_schedule(lr)
        self._p = get_schedule(p)
        self._k = get_schedule(k)

        self._strategy = None
        self._sampled_strategy = None
        self._perturbation = None
        self._last_p = None
        self._baseline = None
        self._commitment = None

        self._counter = 0
        self._interval = 0

    def _sample(self):
        perturbation = self._rng.integers(0, 2, size=self._space.shape)
        perturbation = 2 * perturbation - 1
        self._perturbation = torch.tensor(perturbation,
                                          requires_grad=False,
                                          dtype=torch.float32,
                                          device=self._device)

        self._last_p = self._p.step()
        self._sampled_strategy = self._strategy + self._last_p * self._perturbation
    
    def reset(self):
        initial = self._space.sample(self._rng)
        self._strategy = torch.tensor(initial, 
                                      requires_grad=False, 
                                      dtype=torch.float,
                                      device=self._device)

        self._sample()
        self._counter = 0
        self._interval = 0
        self._baseline = 0
        self._commitment = self._k.step()

        return self._sampled_strategy

    def step(self, strategies):
        self._counter += 1
        if self._counter >= self._commitment:
            self._commitment = self._k.step()
            self._interval += 1

            detached = []
            for strategy in strategies:
                detached.append(strategy.detach())

            payoffs = self._game.payoffs(*detached)

            if self._interval >= self._burn_in:
                estimate = payoffs[self._player_id] - self._baseline
                grad = (estimate / self._last_p) * self._perturbation  # This could become extremely large

                self._strategy.add_(grad, alpha=self._lr.step())

                # Test whether we are hitting NaN or (inf,-inf)
                if any(torch.isnan(self._strategy)):
                    raise ValueError(f"NaN value encountered (strategy: {self._strategy}, "
                                     + f"grad: {grad}, estimate: {estimate}, "
                                     + f"payoff: {payoffs[self._player_id]}, "
                                     + f"baseline: {self._baseline}, last p: {self._last_p})")
                elif any(torch.isinf(self._strategy)):
                    raise ValueError(f"Infinite value encountered (strategy: {self._strategy}, "
                                     + f"grad: {grad}, estimate: {estimate}, "
                                     + f"payoff: {payoffs[self._player_id]}, "
                                     + f"baseline: {self._baseline}, last p: {self._last_p})")

            self._sample()
            self._counter = 0

            self._baseline *= self._baseline_lambda
            self._baseline += (1. - self._baseline_lambda) * payoffs[self._player_id]

        return self._sampled_strategy
