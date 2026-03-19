# cython: language_level=3
"""Game of Y board: triangular hex grid with win detection — Cython."""

from libc.string cimport memcpy, memset


cdef class Board:
    def __init__(self, int size):
        self.size = size
        memset(self._cells, 0, sizeof(self._cells))

    cdef inline int _idx(self, int r, int c):
        return (r << 5) | c  # r * 32 + c

    cpdef Board copy(self):
        cdef Board b = Board.__new__(Board)
        b.size = self.size
        memcpy(b._cells, self._cells, sizeof(self._cells))
        return b

    cpdef bint valid(self, int r, int c):
        return 0 <= r < self.size and 0 <= c <= r

    cpdef list empty_cells(self):
        cdef list result = []
        cdef int r, c
        for r in range(self.size):
            for c in range(r + 1):
                if self._cells[self._idx(r, c)] == 0:
                    result.append((r, c))
        return result

    cpdef place(self, int r, int c, int player):
        self._cells[self._idx(r, c)] = player

    cpdef int get_cell(self, int r, int c):
        return self._cells[self._idx(r, c)]

    cpdef bint check_win(self, int player):
        """BFS: does player have a connected component touching all 3 sides?"""
        cdef int queue_r[256]
        cdef int queue_c[256]
        cdef bint visited[1024]
        cdef int sides, head, tail, cr, cc, nr, nc, d, idx, sr, sc
        cdef int dr[6]
        cdef int dc[6]

        dr[0] = -1; dr[1] = -1; dr[2] = 0; dr[3] = 0; dr[4] = 1; dr[5] = 1
        dc[0] = -1; dc[1] = 0;  dc[2] = -1; dc[3] = 1; dc[4] = 0; dc[5] = 1

        memset(visited, 0, sizeof(visited))

        for sr in range(self.size):
            for sc in range(sr + 1):
                idx = self._idx(sr, sc)
                if self._cells[idx] != player or visited[idx]:
                    continue

                sides = 0
                head = 0
                tail = 0
                queue_r[tail] = sr
                queue_c[tail] = sc
                tail += 1

                while head < tail:
                    cr = queue_r[head]
                    cc = queue_c[head]
                    head += 1
                    idx = self._idx(cr, cc)
                    if visited[idx]:
                        continue
                    visited[idx] = 1

                    if cc == 0:
                        sides |= 1   # left edge
                    if cc == cr:
                        sides |= 2   # right edge
                    if cr == self.size - 1:
                        sides |= 4   # bottom edge
                    if sides == 7:
                        return True

                    for d in range(6):
                        nr = cr + dr[d]
                        nc = cc + dc[d]
                        if 0 <= nr < self.size and 0 <= nc <= nr:
                            if not visited[self._idx(nr, nc)] and self._cells[self._idx(nr, nc)] == player:
                                queue_r[tail] = nr
                                queue_c[tail] = nc
                                tail += 1

        return False

    def _side_membership(self, int r, int c):
        """Python-accessible helper kept for compatibility."""
        sides = set()
        if c == 0:
            sides.add(0)
        if c == r:
            sides.add(1)
        if r == self.size - 1:
            sides.add(2)
        return sides
