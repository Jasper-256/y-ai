"""MCTS agent with UCB1 selection, random rollout."""

import math
import random

from game import Game

# ── Hyperparameters ──
UCB1_C = 1.41           # Exploration constant for UCB1
DEFAULT_ITERATIONS = 8000 # Simulations per move


class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = []
        self.untried_moves = game.legal_moves()
        random.shuffle(self.untried_moves)
        self.visits = 0
        self.wins = 0

    def ucb1(self, c=UCB1_C):
        if self.visits == 0:
            return float("inf")
        return self.wins / self.visits + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def select_child(self):
        return max(self.children, key=lambda ch: ch.ucb1())

    def expand(self):
        move = self.untried_moves.pop()
        child_game = self.game.copy()
        child_game.make_move(*move)
        child = MCTSNode(child_game, parent=self, move=move)
        self.children.append(child)
        return child

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        return self.game.is_over()


def rollout(game):
    """Random playout to game end, return winner."""
    g = game.copy()
    while not g.is_over():
        moves = g.legal_moves()
        if not moves:
            break
        move = random.choice(moves)
        g.make_move(*move)
    return g.winner


def backpropagate(node, winner):
    while node is not None:
        node.visits += 1
        # Win counted from perspective of the player who just moved to reach this node
        if node.parent is not None:
            # The player who made the move leading to this node
            player_who_moved = node.move[2] if len(node.move) == 3 else (
                3 - node.game.current_player if not node.game.is_over() else node.game.winner
            )
            # Actually, simpler: the player who moved is the opposite of current_player
            # (unless game is over). We track via game state.
        # Count win for the node if the winner matches the player who created this node
        if node.move is not None:
            # Player who made the move to get here
            mover = 3 - node.game.current_player if not node.game.is_over() else node.game.winner
            if node.game.is_over():
                mover = node.game.winner
            else:
                mover = 3 - node.game.current_player
            if mover == winner:
                node.wins += 1
        node = node.parent


class MCTSAgent:
    def __init__(self, iterations=DEFAULT_ITERATIONS):
        self.iterations = iterations

    def choose_move(self, game):
        root = MCTSNode(game.copy())

        for _ in range(self.iterations):
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
        best = max(root.children, key=lambda ch: ch.visits)
        return (best.move[0], best.move[1])
