# cython: language_level=3
"""Game wrapper: turn tracking, move application, serialization — Cython."""

from board cimport Board


cdef class Game:
    def __init__(self, int size=9):
        self.board = Board(size)
        self.current_player = 1  # Red goes first
        self.winner = 0
        self.move_history = []

    cpdef Game copy(self):
        cdef Game g = Game.__new__(Game)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.winner = self.winner
        g.move_history = list(self.move_history)
        return g

    cpdef list legal_moves(self):
        if self.winner:
            return []
        return self.board.empty_cells()

    cpdef bint make_move(self, int row, int col):
        if self.winner:
            return False
        if self.board.get_cell(row, col) != 0:
            return False
        self.board.place(row, col, self.current_player)
        self.move_history.append((row, col, self.current_player))
        if self.board.check_win(self.current_player):
            self.winner = self.current_player
        else:
            self.current_player = 3 - self.current_player
        return True

    cpdef bint is_over(self):
        return self.winner != 0

    def to_dict(self):
        cdef int r, c
        cells = {}
        for r in range(self.board.size):
            for c in range(r + 1):
                cells[f"{r},{c}"] = self.board.get_cell(r, c)
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
