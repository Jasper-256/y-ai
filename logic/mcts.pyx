# cython: language_level=3
"""MCTS agent with UCB1 selection, random rollout — Cython implementation."""

from libc.math cimport sqrt, log
import random

from game cimport Game


# ── Hyperparameters ──
cdef double UCB1_C = 1.41
cdef int DEFAULT_ITERATIONS = 1_000


cdef class MCTSNode:
    cdef Game game
    cdef MCTSNode parent
    cdef public tuple move
    cdef public list children
    cdef public list untried_moves
    cdef public int visits
    cdef public double wins

    def __init__(self, Game game, MCTSNode parent=None, tuple move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = []
        self.untried_moves = game.legal_moves()
        random.shuffle(self.untried_moves)
        self.visits = 0
        self.wins = 0.0

    cdef double ucb1(self):
        if self.visits == 0:
            return float("inf")
        return self.wins / self.visits + UCB1_C * sqrt(
            log(self.parent.visits) / self.visits
        )

    cdef MCTSNode select_child(self):
        cdef MCTSNode best = None
        cdef double best_val = -1.0
        cdef double val
        cdef MCTSNode ch
        for ch in self.children:
            val = ch.ucb1()
            if val > best_val:
                best_val = val
                best = ch
        return best

    cdef MCTSNode expand(self):
        cdef tuple move = <tuple>self.untried_moves.pop()
        cdef Game child_game = self.game.copy()
        child_game.make_move(<int>move[0], <int>move[1])
        cdef MCTSNode child = MCTSNode(child_game, parent=self, move=move)
        self.children.append(child)
        return child

    cdef bint is_fully_expanded(self):
        return len(self.untried_moves) == 0

    cdef bint is_terminal(self):
        return self.game.is_over()


cdef int rollout(Game game):
    """Random playout to game end, return winner."""
    cdef Game g = game.copy()
    cdef list moves
    cdef tuple move
    while not g.is_over():
        moves = g.legal_moves()
        if not moves:
            break
        move = <tuple>random.choice(moves)
        g.make_move(<int>move[0], <int>move[1])
    return g.winner


cdef void backpropagate(MCTSNode node, int winner):
    cdef int mover
    while node is not None:
        node.visits += 1
        if node.move is not None:
            if node.game.is_over():
                mover = node.game.winner
            else:
                mover = 3 - node.game.current_player
            if mover == winner:
                node.wins += 1.0
        node = node.parent


class MCTSAgent:
    def __init__(self, int iterations=DEFAULT_ITERATIONS):
        self.iterations = iterations

    def analyze(self, game):
        """Return root child stats as (move, visits, wins) after MCTS search.

        ``wins`` is from the perspective of the player who made ``move``. At
        the root, that is the current player, so these stats are useful as a
        policy distribution and value target for teacher training.
        """
        cdef MCTSNode root = MCTSNode((<Game>game).copy())
        cdef MCTSNode node
        cdef int winner
        cdef int i
        cdef MCTSNode ch
        cdef list stats = []

        for i in range(self.iterations):
            node = root

            while node.is_fully_expanded() and node.children:
                node = node.select_child()

            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            winner = rollout(node.game)
            backpropagate(node, winner)

        for ch in root.children:
            stats.append((ch.move, ch.visits, ch.wins))
        return stats

    def choose_move(self, game):
        cdef MCTSNode root = MCTSNode((<Game>game).copy())
        cdef MCTSNode node
        cdef int winner
        cdef int i

        for i in range(self.iterations):
            node = root

            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.select_child()

            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # Rollout
            winner = rollout(node.game)

            # Backpropagation
            backpropagate(node, winner)

        # Choose move with most visits
        if not root.children:
            return None
        cdef MCTSNode best = None
        cdef int best_visits = -1
        cdef MCTSNode ch
        for ch in root.children:
            if ch.visits > best_visits:
                best_visits = ch.visits
                best = ch
        return (best.move[0], best.move[1])
