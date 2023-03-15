from hi_c.games.matrix import IteratedGame, MatrixGame
from hi_c.games.polynomial import TandemGame, HamiltonianGame

GAMES = {
    "iterated": IteratedGame,
    "matrix": MatrixGame,
    "tandem": TandemGame,
    "hamiltonian": HamiltonianGame,
}

def get_game_class(name):
    if name not in GAMES:
        raise ValueError(f"Game '{name}' is not defined")
    
    return GAMES[name]


__all__ = []