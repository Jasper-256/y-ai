"""Random agent — picks a uniformly random legal move."""

import random


class RandomAgent:
    def choose_move(self, game):
        moves = game.legal_moves()
        if not moves:
            return None
        return random.choice(moves)
