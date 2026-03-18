"""Game of Y board: triangular hex grid with win detection."""

from collections import deque


class Board:
    def __init__(self, size=9):
        self.size = size
        # cells[(r, c)] = 0 (empty), 1 (Red), 2 (Blue)
        self.cells = {}
        for r in range(size):
            for c in range(r + 1):
                self.cells[(r, c)] = 0

    def copy(self):
        b = Board.__new__(Board)
        b.size = self.size
        b.cells = dict(self.cells)
        return b

    def valid(self, r, c):
        return 0 <= r < self.size and 0 <= c <= r

    def neighbors(self, r, c):
        for dr, dc in [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)]:
            nr, nc = r + dr, c + dc
            if self.valid(nr, nc):
                yield (nr, nc)

    def empty_cells(self):
        return [pos for pos, v in self.cells.items() if v == 0]

    def place(self, r, c, player):
        self.cells[(r, c)] = player

    def _side_membership(self, r, c):
        """Return set of sides (0=left, 1=right, 2=bottom) this cell belongs to."""
        sides = set()
        if c == 0:
            sides.add(0)       # left edge
        if c == r:
            sides.add(1)       # right edge
        if r == self.size - 1:
            sides.add(2)       # bottom edge
        return sides

    def check_win(self, player):
        """BFS: does player have a connected component touching all 3 sides?"""
        player_cells = [pos for pos, v in self.cells.items() if v == player]
        if not player_cells:
            return False

        visited = set()
        # Try BFS from each unvisited player cell
        for start in player_cells:
            if start in visited:
                continue
            sides_touched = set()
            queue = deque([start])
            component = set()
            while queue:
                pos = queue.popleft()
                if pos in component:
                    continue
                component.add(pos)
                sides_touched |= self._side_membership(*pos)
                if len(sides_touched) == 3:
                    return True
                for nb in self.neighbors(*pos):
                    if nb not in component and self.cells.get(nb) == player:
                        queue.append(nb)
            visited |= component
        return False
