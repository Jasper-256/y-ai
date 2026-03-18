"""Game wrapper: turn tracking, move application, serialization."""

from board import Board


class Game:
    def __init__(self, size=9):
        self.board = Board(size)
        self.current_player = 1  # Red goes first
        self.winner = 0
        self.move_history = []

    def copy(self):
        g = Game.__new__(Game)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.winner = self.winner
        g.move_history = list(self.move_history)
        return g

    def legal_moves(self):
        if self.winner:
            return []
        return self.board.empty_cells()

    def make_move(self, row, col):
        if self.winner:
            return False
        if self.board.cells.get((row, col), -1) != 0:
            return False
        self.board.place(row, col, self.current_player)
        self.move_history.append((row, col, self.current_player))
        if self.board.check_win(self.current_player):
            self.winner = self.current_player
        else:
            self.current_player = 3 - self.current_player
        return True

    def is_over(self):
        return self.winner != 0

    def to_dict(self):
        cells = {}
        for (r, c), v in self.board.cells.items():
            cells[f"{r},{c}"] = v
        return {
            "size": self.board.size,
            "cells": cells,
            "current_player": self.current_player,
            "winner": self.winner,
            "move_history": [
                {"row": r, "col": c, "player": p}
                for r, c, p in self.move_history
            ],
        }
