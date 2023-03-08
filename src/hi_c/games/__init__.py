from hi_c.games.iterated import IteratedGame
from hi_c.games.polynomial import TandemGame, HamiltonianGame

GAMES = {
    "iterated": IteratedGame,
    "tandem": TandemGame,
    "hamiltonian": HamiltonianGame,
}

def get_game_class(name):
    if name not in GAMES:
        raise ValueError(f"Game '{name}' is not defined")
    
    return GAMES[name]


__all__ = []