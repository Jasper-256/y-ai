# cython: language_level=3
cdef class Board:
    cdef public int size
    cdef int _cells[1024]  # indexed by r * 32 + c

    cdef inline int _idx(self, int r, int c)
    cpdef Board copy(self)
    cpdef bint valid(self, int r, int c)
    cpdef list empty_cells(self)
    cpdef place(self, int r, int c, int player)
    cpdef int get_cell(self, int r, int c)
    cpdef bint check_win(self, int player)
