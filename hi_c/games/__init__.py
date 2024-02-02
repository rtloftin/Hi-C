from hi_c.games.matrix import MatrixGame, IteratedMatrixGame
from hi_c.games.polynomial import TandemGame, HamiltonianGame
from hi_c.games.optimization import Quadratic, Gaussian
from hi_c.games.cournot import Cournot

GAMES = {
    "iterated": IteratedMatrixGame,
    "matrix": MatrixGame,
    "tandem": TandemGame,
    "hamiltonian": HamiltonianGame,
    "quadratic": Quadratic,
    "gaussian": Gaussian,
    "cournot": Cournot,
}


def get_game_class(name):
    if name not in GAMES:
        raise ValueError(f"Game '{name}' is not defined")
    
    return GAMES[name]


__all__ = [get_game_class]
