"""TD(λ) agent with eligibility traces and a small MLP value function for Game of Y.

Like TD(0), the agent learns a state-value function V(s) from the perspective
of the current player, using 1-ply lookahead for move selection.

The difference from TD(0) is the use of eligibility traces.  Instead of
updating only the most recently visited state, TD(λ) maintains a decaying
trace of past gradients so that a single TD error propagates credit backward
across many states:

    e(t) = γλ · e(t-1) + ∇V(s_t)
    δ(t) = reward + γ · V(s_{t+1}) - V(s_t)
    w   <- w + α · δ(t) · e(t)

With γ=1 and reward=0 for non-terminal states this simplifies to:
    e(t) = λ · e(t-1) + ∇V(s_t)
    δ(t) = V(s_{t+1}) - V(s_t)          (non-terminal)
    δ(t) = outcome   - V(s_t)           (terminal)
    w   <- w + α · δ · e(t)

λ=0 recovers TD(0); λ=1 approximates Monte-Carlo.
"""

import os
import random
import pickle
import numpy as np

from training import train

# ── Feature extraction ──────────────────────────────────────────────────────

def _board_features(game):
    """Encode the board as a feature vector from the current player's perspective.

    For each cell on the triangular grid:
      - 3 one-hot values: empty, mine, opponent's
    Plus one bias feature (always 1).
    """
    board = game.board
    size = board.size
    me = game.current_player
    opp = 3 - me

    feats = []
    for r in range(size):
        for c in range(r + 1):
            v = board.get_cell(r, c)
            feats.append(1.0 if v == 0 else 0.0)   # empty
            feats.append(1.0 if v == me else 0.0)   # mine
            feats.append(1.0 if v == opp else 0.0)  # opponent
    feats.append(1.0)  # bias
    return np.array(feats, dtype=np.float64)


def feature_size(board_size):
    """Number of features for a given board size."""
    n_cells = board_size * (board_size + 1) // 2
    return n_cells * 3 + 1  # 3 per cell + bias


# ── Simple MLP with eligibility traces ────────────────────────────────────

def _relu(x):
    return np.maximum(x, 0.0)


def _relu_deriv(x):
    return (x > 0).astype(np.float64)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class MLP:
    """Two-layer MLP: input -> hidden (ReLU) -> 1 (sigmoid), with eligibility traces."""

    def __init__(self, input_size, hidden_size=128):
        scale1 = np.sqrt(2.0 / input_size)
        scale2 = np.sqrt(2.0 / hidden_size)
        self.W1 = np.random.randn(hidden_size, input_size) * scale1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(1, hidden_size) * scale2
        self.b2 = np.zeros(1)
        # Eligibility traces (same shape as weights)
        self.reset_traces()

    def reset_traces(self):
        self.eW1 = np.zeros_like(self.W1)
        self.eb1 = np.zeros_like(self.b1)
        self.eW2 = np.zeros_like(self.W2)
        self.eb2 = np.zeros_like(self.b2)

    def forward(self, x):
        """Return (value, cache) where value is in [0, 1]."""
        z1 = self.W1 @ x + self.b1
        a1 = _relu(z1)
        z2 = self.W2 @ a1 + self.b2
        value = _sigmoid(z2[0])
        cache = (x, z1, a1, z2)
        return value, cache

    def backward(self, cache, grad_output):
        """Compute gradients w.r.t. all parameters."""
        x, z1, a1, z2 = cache
        v = _sigmoid(z2[0])
        dsig = v * (1.0 - v) * grad_output

        dW2 = dsig * a1.reshape(1, -1)
        db2 = np.array([dsig])

        da1 = dsig * self.W2[0]
        dz1 = da1 * _relu_deriv(z1)

        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def update_traces(self, grads, lam):
        """Decay old traces by -λ and accumulate new gradients.

        The negation accounts for alternating perspectives in a two-player
        game: the value function switches viewpoint each turn, so past
        gradients must be negated when carried forward.
        """
        self.eW1 = -lam * self.eW1 + grads["W1"]
        self.eb1 = -lam * self.eb1 + grads["b1"]
        self.eW2 = -lam * self.eW2 + grads["W2"]
        self.eb2 = -lam * self.eb2 + grads["b2"]

    def apply_td_update(self, td_error, lr):
        """Apply TD update using eligibility traces."""
        self.W1 += lr * td_error * self.eW1
        self.b1 += lr * td_error * self.eb1
        self.W2 += lr * td_error * self.eW2
        self.b2 += lr * td_error * self.eb2


# ── TD(λ) Agent ────────────────────────────────────────────────────────────

class TDLambdaAgent:
    """TD(λ) agent with eligibility traces and 1-ply lookahead.

    Args:
        board_size: Size of the board (needed for feature dimensions).
        hidden_size: Hidden layer width.
        lr: Learning rate (alpha).
        lam: Lambda — trace decay parameter in [0, 1].
        epsilon: Exploration rate for epsilon-greedy during training.
    """

    def __init__(self, board_size=7, hidden_size=128, lr=0.01, lam=0.7, epsilon=0.1):
        self.board_size = board_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.lam = lam
        self.epsilon = epsilon
        n_feat = feature_size(board_size)
        self.net = MLP(n_feat, hidden_size)
        self.training = False
        # Trace for TD updates during a game
        self._prev_cache = None
        self._prev_value = None

    def _evaluate(self, game):
        """Return V(s) from perspective of game.current_player."""
        feats = _board_features(game)
        value, cache = self.net.forward(feats)
        return value, cache

    def choose_move(self, game):
        """Pick a move via 1-ply lookahead (or epsilon-greedy if training)."""
        moves = game.legal_moves()
        if not moves:
            return None

        # Epsilon-greedy exploration during training
        if self.training and random.random() < self.epsilon:
            move = random.choice(moves)
            if self.training:
                self._td_step(game)
            return move

        best_move = None
        best_value = -1.0

        for move in moves:
            child = game.copy()
            child.make_move(move[0], move[1])

            if child.is_over():
                # Winning move — take it immediately
                if child.winner == game.current_player:
                    if self.training:
                        self._td_step(game, terminal_outcome=1.0)
                    return move
                continue  # skip losing moves if possible

            # Evaluate from opponent's perspective, invert
            opp_value, _ = self._evaluate(child)
            my_value = 1.0 - opp_value

            if my_value > best_value:
                best_value = my_value
                best_move = move

        if best_move is None:
            best_move = moves[0]

        if self.training:
            self._td_step(game)

        return best_move

    def _td_step(self, game, terminal_outcome=None):
        """Perform a TD(λ) weight update using eligibility traces."""
        current_value, current_cache = self._evaluate(game)

        if self._prev_cache is not None:
            # Target: opponent's perspective inverted
            target = 1.0 - current_value
            td_error = target - self._prev_value

            # Accumulate previous state's gradients into traces
            # Pass 1.0 as grad_output to get raw ∇V w.r.t. weights
            grads = self.net.backward(self._prev_cache, 1.0)
            self.net.update_traces(grads, self.lam)
            # Update all weights proportional to traces
            self.net.apply_td_update(td_error, self.lr)

        if terminal_outcome is not None:
            td_error = terminal_outcome - current_value
            grads = self.net.backward(current_cache, 1.0)
            self.net.update_traces(grads, self.lam)
            self.net.apply_td_update(td_error, self.lr)
            self._prev_cache = None
            self._prev_value = None
            self.net.reset_traces()
        else:
            self._prev_cache = current_cache
            self._prev_value = current_value

    def end_game(self, game):
        """Call at game end to do the final TD update."""
        if not self.training or self._prev_cache is None:
            return
        if game.winner == 0:
            target = 0.5
        else:
            target = 0.0  # lost
        td_error = target - self._prev_value
        grads = self.net.backward(self._prev_cache, 1.0)
        self.net.update_traces(grads, self.lam)
        self.net.apply_td_update(td_error, self.lr)
        self._prev_cache = None
        self._prev_value = None
        self.net.reset_traces()

    def train(self, num_games=1000, opponent=None, board_size=None):
        """Train via self-play or against a given opponent.

        Args:
            num_games: Number of training games.
            opponent: Another agent to train against. If None, trains via
                      self-play (plays both sides).
            board_size: Override board size for training games.
        """
        def reset():
            self._prev_cache = None
            self._prev_value = None
            self.net.reset_traces()
        
        def update(td_error):
            grads = self.net.backward(self._prev_cache, 1.0)
            self.net.update_traces(grads, self.lam)
            self.net.apply_td_update(td_error, self.lr)
        
        train(self, reset, update, num_games, opponent, board_size)

        print(f"Training complete ({num_games} games, λ={self.lam}).")

    def save(self, path):
        """Save model weights to a file."""
        data = {
            "board_size": self.board_size,
            "hidden_size": self.hidden_size,
            "lr": self.lr,
            "lam": self.lam,
            "epsilon": self.epsilon,
            "W1": self.net.W1,
            "b1": self.net.b1,
            "W2": self.net.W2,
            "b2": self.net.b2,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        """Load a previously saved model."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            board_size=data["board_size"],
            hidden_size=data["hidden_size"],
            lr=data["lr"],
            lam=data["lam"],
            epsilon=data["epsilon"],
        )
        agent.net.W1 = data["W1"]
        agent.net.b1 = data["b1"]
        agent.net.W2 = data["W2"]
        agent.net.b2 = data["b2"]
        return agent
