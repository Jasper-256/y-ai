"""TD(0) agent with a CNN value function for Game of Y.

Mirrors ``td_agent.py`` but replaces the flat MLP value network with a small
convolutional network that operates on a 2D embedding of the triangular
board.

Board embedding: cell ``(r, c)`` (with ``c ≤ r``) is placed at position
``(r, c)`` of an ``N × N`` square grid — the lower-left triangle. Positions
with ``c > r`` are off-board. Since a 2D CNN has no way to tell which
positions are real board cells, one of the input channels is a binary
"exists" mask. This diagonal layout preserves hex adjacency within a 3×3
kernel: six of the eight neighbour offsets correspond to real hex
neighbours of ``(r, c)``, so a 3×3 conv can learn local board
interactions. The layout breaks the triangle's three-way symmetry, but the
CNN just learns anisotropic filters.

Network: input ``(4, N, N)`` → Conv(32) → Conv(32) → Conv(16) → flatten
→ Dense(64) → Dense(1, sigmoid). All convs are 3×3 with padding 1 and
ReLU activations. Output ``V(s) ∈ [0, 1]``.

Move selection and TD(0) updates match ``TDAgent``: 1-ply lookahead picks
the child that maximises ``1 - V(child)`` (child is evaluated from the
opponent's perspective), and training uses self-play TD(0).
"""

import random
import pickle
import numpy as np
from numpy.lib.stride_tricks import as_strided

from training import train

# ── Feature extraction ──────────────────────────────────────────────────────

N_CHANNELS = 4  # exists, mine, opp, empty


def _board_features(game):
    """Encode the board as a ``(4, N, N)`` tensor from the mover's perspective.

    Channels:
      0: exists — 1 if ``(r, c)`` is inside the triangle (``c ≤ r``)
      1: mine — 1 if my stone
      2: opp — 1 if opponent's stone
      3: empty — 1 if empty valid cell
    """
    board = game.board
    size = board.size
    me = game.current_player
    opp = 3 - me
    x = np.zeros((N_CHANNELS, size, size), dtype=np.float64)
    for r in range(size):
        for c in range(r + 1):
            x[0, r, c] = 1.0
            v = board.get_cell(r, c)
            if v == me:
                x[1, r, c] = 1.0
            elif v == opp:
                x[2, r, c] = 1.0
            else:
                x[3, r, c] = 1.0
    return x


# ── Numpy CNN primitives ────────────────────────────────────────────────────

def _relu(x):
    return np.maximum(x, 0.0)


def _relu_deriv(x):
    return (x > 0).astype(np.float64)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _im2col(x, kh, kw, pad):
    """``x: (C, H, W)`` → ``(C*kh*kw, out_h*out_w)`` via sliding windows."""
    C, H, W = x.shape
    x_padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
    out_h = H + 2 * pad - kh + 1
    out_w = W + 2 * pad - kw + 1
    s_c, s_h, s_w = x_padded.strides
    shape = (C, kh, kw, out_h, out_w)
    strides = (s_c, s_h, s_w, s_h, s_w)
    windows = as_strided(x_padded, shape=shape, strides=strides)
    return np.ascontiguousarray(windows.reshape(C * kh * kw, out_h * out_w))


def _col2im(cols, C, H, W, kh, kw, pad):
    """Inverse of ``_im2col``: scatter column contributions back to ``(C, H, W)``."""
    out_h = H + 2 * pad - kh + 1
    out_w = W + 2 * pad - kw + 1
    padded = np.zeros((C, H + 2 * pad, W + 2 * pad), dtype=np.float64)
    cols_r = cols.reshape(C, kh, kw, out_h, out_w)
    for i in range(kh):
        for j in range(kw):
            padded[:, i:i + out_h, j:j + out_w] += cols_r[:, i, j, :, :]
    if pad > 0:
        return padded[:, pad:-pad, pad:-pad]
    return padded


# ── CNN value network ──────────────────────────────────────────────────────

class CNN:
    """[conv 3×3 + ReLU] × depth → flatten → dense(ReLU) → dense(sigmoid).

    Every conv uses ``padding=1`` so spatial dims are preserved. Output is a
    scalar value in ``[0, 1]``.
    """

    def __init__(self, board_size=7, conv_channels=(32, 32, 16), fc_hidden=64):
        self.board_size = board_size
        self.conv_channels = tuple(conv_channels)
        self.fc_hidden = fc_hidden

        self.Wc = []
        self.bc = []
        prev = N_CHANNELS
        for c in conv_channels:
            scale = np.sqrt(2.0 / (prev * 9))
            self.Wc.append(np.random.randn(c, prev, 3, 3) * scale)
            self.bc.append(np.zeros(c))
            prev = c

        flat_size = prev * board_size * board_size
        scale = np.sqrt(2.0 / flat_size)
        self.Wf1 = np.random.randn(fc_hidden, flat_size) * scale
        self.bf1 = np.zeros(fc_hidden)
        scale = np.sqrt(2.0 / fc_hidden)
        self.Wf2 = np.random.randn(1, fc_hidden) * scale
        self.bf2 = np.zeros(1)

    def forward(self, x):
        """Return ``(value, cache)``. ``x`` has shape ``(4, N, N)``."""
        cache = {"conv": []}
        h = x
        for W, b in zip(self.Wc, self.bc):
            cols = _im2col(h, 3, 3, 1)
            W_flat = W.reshape(W.shape[0], -1)
            z = W_flat @ cols + b[:, None]
            z = z.reshape(W.shape[0], h.shape[1], h.shape[2])
            cache["conv"].append((h.shape, cols, z))
            h = _relu(z)
        cache["flat_shape"] = h.shape
        flat = h.reshape(-1)
        cache["flat"] = flat
        z1 = self.Wf1 @ flat + self.bf1
        a1 = _relu(z1)
        cache["z1"] = z1
        cache["a1"] = a1
        z2 = self.Wf2 @ a1 + self.bf2
        v = _sigmoid(z2[0])
        cache["value"] = v
        return v, cache

    def backward(self, cache, grad_output):
        """Backprop. ``grad_output`` is ``dL/dvalue`` (a scalar)."""
        v = cache["value"]
        dsig = v * (1.0 - v) * grad_output

        dWf2 = dsig * cache["a1"].reshape(1, -1)
        dbf2 = np.array([dsig])
        da1 = dsig * self.Wf2[0]

        dz1 = da1 * _relu_deriv(cache["z1"])
        dWf1 = np.outer(dz1, cache["flat"])
        dbf1 = dz1
        dflat = self.Wf1.T @ dz1
        dh = dflat.reshape(cache["flat_shape"])

        dWc = [None] * len(self.Wc)
        dbc = [None] * len(self.bc)
        for i in range(len(self.Wc) - 1, -1, -1):
            in_shape, cols, z = cache["conv"][i]
            dz = dh * _relu_deriv(z)
            dz_flat = dz.reshape(dz.shape[0], -1)
            W = self.Wc[i]
            C_out, C_in, kh, kw = W.shape
            W_flat = W.reshape(C_out, -1)
            dW_flat = dz_flat @ cols.T
            dWc[i] = dW_flat.reshape(C_out, C_in, kh, kw)
            dbc[i] = dz_flat.sum(axis=1)
            if i > 0:
                dcols = W_flat.T @ dz_flat
                dh = _col2im(dcols, C_in, in_shape[1], in_shape[2], kh, kw, 1)

        return {
            "Wc": dWc,
            "bc": dbc,
            "Wf1": dWf1,
            "bf1": dbf1,
            "Wf2": dWf2,
            "bf2": dbf2,
        }

    def apply_grads(self, grads, lr):
        for i in range(len(self.Wc)):
            self.Wc[i] += lr * grads["Wc"][i]
            self.bc[i] += lr * grads["bc"][i]
        self.Wf1 += lr * grads["Wf1"]
        self.bf1 += lr * grads["bf1"]
        self.Wf2 += lr * grads["Wf2"]
        self.bf2 += lr * grads["bf2"]


# ── TD(0) CNN Agent ────────────────────────────────────────────────────────

class TDCNNAgent:
    """TD(0) agent with a CNN value function and 1-ply lookahead.

    Args:
        board_size: Size of the board (needed for spatial dimensions).
        conv_channels: Channels per conv layer (depth = len).
        fc_hidden: Size of the fully-connected hidden layer.
        lr: Learning rate (alpha).
        epsilon: Exploration rate for epsilon-greedy during training.
    """

    def __init__(self, board_size=7, conv_channels=(32, 32, 16), fc_hidden=64,
                 lr=0.005, epsilon=0.1):
        self.board_size = board_size
        self.conv_channels = tuple(conv_channels)
        self.fc_hidden = fc_hidden
        self.lr = lr
        self.epsilon = epsilon
        self.net = CNN(board_size, self.conv_channels, fc_hidden)
        self.training = False
        self._prev_cache = None
        self._prev_value = None

    def _evaluate(self, game):
        """Return V(s) from perspective of ``game.current_player``."""
        feats = _board_features(game)
        value, cache = self.net.forward(feats)
        return value, cache

    def choose_move(self, game):
        """Pick a move via 1-ply lookahead (or epsilon-greedy if training)."""
        moves = game.legal_moves()
        if not moves:
            return None

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
                if child.winner == game.current_player:
                    if self.training:
                        self._td_step(game, terminal_outcome=1.0)
                    return move
                continue

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
        """Perform a TD(0) weight update."""
        current_value, current_cache = self._evaluate(game)

        if self._prev_cache is not None:
            target = 1.0 - current_value
            td_error = target - self._prev_value
            grads = self.net.backward(self._prev_cache, td_error)
            self.net.apply_grads(grads, self.lr)

        if terminal_outcome is not None:
            td_error = terminal_outcome - current_value
            grads = self.net.backward(current_cache, td_error)
            self.net.apply_grads(grads, self.lr)
            self._prev_cache = None
            self._prev_value = None
        else:
            self._prev_cache = current_cache
            self._prev_value = current_value

    def reset_episode(self):
        """Clear TD state at episode boundaries (prev state)."""
        self._prev_cache = None
        self._prev_value = None

    def update(self, td_error):
        """Apply one TD weight step from the training loop's terminal error."""
        grads = self.net.backward(self._prev_cache, td_error)
        self.net.apply_grads(grads, self.lr)

    def train(self, num_games=1000, opponent=None, board_size=None, checkpoints=None):
        """Train via self-play or against a given opponent."""
        train(self, num_games, opponent, board_size, checkpoints)
        print(f"Training complete ({num_games} games).")

    def save(self, path):
        """Save model weights to a file."""
        data = {
            "board_size": self.board_size,
            "conv_channels": self.conv_channels,
            "fc_hidden": self.fc_hidden,
            "lr": self.lr,
            "epsilon": self.epsilon,
            "Wc": self.net.Wc,
            "bc": self.net.bc,
            "Wf1": self.net.Wf1,
            "bf1": self.net.bf1,
            "Wf2": self.net.Wf2,
            "bf2": self.net.bf2,
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
            conv_channels=tuple(data["conv_channels"]),
            fc_hidden=data["fc_hidden"],
            lr=data["lr"],
            epsilon=data["epsilon"],
        )
        agent.net.Wc = data["Wc"]
        agent.net.bc = data["bc"]
        agent.net.Wf1 = data["Wf1"]
        agent.net.bf1 = data["bf1"]
        agent.net.Wf2 = data["Wf2"]
        agent.net.bf2 = data["bf2"]
        return agent
