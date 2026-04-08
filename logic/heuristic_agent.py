"""Heuristic agent — picks a move by 1-ply lookahead with a static evaluation.

The evaluator prefers positions where your strongest connected group reaches
more board sides (then has more stones); it subtracts the opponent's best
group and lightly rewards having more empty cells adjacent to your stones
than to theirs.
"""

import random

# Hex neighbors on the triangular board (same adjacency as board.pyx).
_DR = (-1, -1, 0, 0, 1, 1)
_DC = (-1, 0, -1, 1, 0, 1)


def _neighbors(board, r, c):
    for i in range(6):
        nr, nc = r + _DR[i], c + _DC[i]
        if board.valid(nr, nc):
            yield nr, nc


def _side_mask(board, r, c):
    m = 0
    if c == 0:
        m |= 1
    if c == r:
        m |= 2
    if r == board.size - 1:
        m |= 4
    return m


def _best_group_key(board, player):
    """Return (num_distinct_sides, stones) of the best connected component."""
    size = board.size
    seen = [[False] * (r + 1) for r in range(size)]
    best = (0, 0)

    for sr in range(size):
        for sc in range(sr + 1):
            if seen[sr][sc] or board.get_cell(sr, sc) != player:
                continue
            stack = [(sr, sc)]
            seen[sr][sc] = True
            sides = 0
            count = 0
            while stack:
                r, c = stack.pop()
                count += 1
                sides |= _side_mask(board, r, c)
                for nr, nc in _neighbors(board, r, c):
                    if seen[nr][nc]:
                        continue
                    v = board.get_cell(nr, nc)
                    if v == player:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            n_sides = sides.bit_count()
            cand = (n_sides, count)
            if cand > best:
                best = cand
    return best


def _frontier_count(board, player):
    """Empty cells touching at least one stone of player."""
    size = board.size
    touched = set()
    for r in range(size):
        for c in range(r + 1):
            if board.get_cell(r, c) != player:
                continue
            for nr, nc in _neighbors(board, r, c):
                if board.get_cell(nr, nc) == 0:
                    touched.add((nr, nc))
    return len(touched)


def evaluate_position(board, perspective_player):
    """Higher is better for perspective_player (non-terminal positions)."""
    opp = 3 - perspective_player
    us = _best_group_key(board, perspective_player)
    them = _best_group_key(board, opp)
    score = (us[0] - them[0]) * 10_000 + (us[1] - them[1])
    score += 0.25 * (_frontier_count(board, perspective_player) - _frontier_count(board, opp))
    return score


class HeuristicAgent:
    def choose_move(self, game):
        moves = game.legal_moves()
        if not moves:
            return None
        me = game.current_player
        best_score = None
        best_moves = []
        for move in moves:
            g = game.copy()
            g.make_move(move[0], move[1])
            if g.winner == me:
                return move
            if g.is_over():
                continue
            s = evaluate_position(g.board, me)
            if best_score is None or s > best_score:
                best_score = s
                best_moves = [move]
            elif s == best_score:
                best_moves.append(move)
        if not best_moves:
            return random.choice(moves)
        return random.choice(best_moves)
