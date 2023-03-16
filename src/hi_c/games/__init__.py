from hi_c.games.matrix import IteratedGame, MatrixGame
from hi_c.games.polynomial import TandemGame, HamiltonianGame
from hi_c.games.optimization import Quadratic, Gaussian

GAMES = {
    "iterated": IteratedGame,
    "matrix": MatrixGame,
    "tandem": TandemGame,
    "hamiltonian": HamiltonianGame,
    "quadratic": Quadratic,
    "gaussian": Gaussian,
}

def get_game_class(name):
    if name not in GAMES:
        raise ValueError(f"Game '{name}' is not defined")
    
    return GAMES[name]


__all__ = []