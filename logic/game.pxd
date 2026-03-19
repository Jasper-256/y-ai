# cython: language_level=3
from board cimport Board

cdef class Game:
    cdef public Board board
    cdef public int current_player
    cdef public int winner
    cdef public list move_history

    cpdef Game copy(self)
    cpdef list legal_moves(self)
    cpdef bint make_move(self, int row, int col)
    cpdef bint is_over(self)
