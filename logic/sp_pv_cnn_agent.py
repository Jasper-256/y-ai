"""Self-play policy/value CNN for PV-MCTS.

This variant learns from its own MCTS searches, like ``sp_pv_mcts``, but uses
a small CNN so it can learn local board patterns. Training examples are
augmented with the six symmetries of the triangular Game-of-Y board.
"""

import math
import os
import pickle
import random
from collections import deque

import numpy as np

from pv_mcts_agent import (
    PVMCTSAgent,
    TrainingExample,
    _cell_index,
    _num_cells,
    _policy_from_stats,
    _sample_move_from_stats,
)
from td_cnn_agent import _col2im, _im2col, _relu, _relu_deriv, _sigmoid


N_CHANNELS = 4  # exists, mine, opponent, empty
SYMMETRIES = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def _coords(row, col, size):
    return col, row - col, size - 1 - row


def _from_coords(coords, size):
    x, y, z = coords
    return size - 1 - z, x


def _transform_cell(row, col, size, perm):
    coords = _coords(row, col, size)
    return _from_coords((coords[perm[0]], coords[perm[1]], coords[perm[2]]), size)


def _board_features(game):
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


def _softmax_masked(logits, legal_indices, temperature=1.0):
    probs = np.zeros_like(logits, dtype=np.float64)
    if not legal_indices:
        return probs
    legal = np.array(legal_indices, dtype=np.int64)
    z = logits[legal] / max(temperature, 1e-6)
    z = z - np.max(z)
    exp_z = np.exp(np.clip(z, -60.0, 60.0))
    total = float(np.sum(exp_z))
    if total <= 0.0 or not np.isfinite(total):
        probs[legal] = 1.0 / len(legal)
    else:
        probs[legal] = exp_z / total
    return probs


def _transform_tensor(x, perm):
    size = x.shape[1]
    y = np.zeros_like(x)
    for r in range(size):
        for c in range(r + 1):
            tr, tc = _transform_cell(r, c, size, perm)
            y[:, tr, tc] = x[:, r, c]
    return y


def _transform_policy(policy, size, perm):
    out = np.zeros_like(policy)
    for r in range(size):
        for c in range(r + 1):
            tr, tc = _transform_cell(r, c, size, perm)
            out[_cell_index(tr, tc)] = policy[_cell_index(r, c)]
    return out


def augment_examples(examples, board_size):
    augmented = []
    for ex in examples:
        for perm in SYMMETRIES:
            augmented.append(
                TrainingExample(
                    features=_transform_tensor(ex.features, perm),
                    policy=_transform_policy(ex.policy, board_size, perm),
                    player=ex.player,
                    winner=ex.winner,
                    value=ex.value,
                )
            )
    return augmented


class PolicyValueCNN:
    """Small CNN with shared trunk, policy head, and value head."""

    def __init__(self, board_size=7, conv_channels=(24, 24, 16), fc_hidden=96, seed=None):
        self.board_size = board_size
        self.conv_channels = tuple(conv_channels)
        self.fc_hidden = fc_hidden
        self.n_cells = _num_cells(board_size)
        rng = np.random.default_rng(seed)

        self.Wc = []
        self.bc = []
        prev = N_CHANNELS
        for channels in self.conv_channels:
            scale = math.sqrt(2.0 / (prev * 9))
            self.Wc.append(rng.normal(0.0, scale, (channels, prev, 3, 3)))
            self.bc.append(np.zeros(channels))
            prev = channels

        flat_size = prev * board_size * board_size
        self.Wf = rng.normal(0.0, math.sqrt(2.0 / flat_size), (fc_hidden, flat_size))
        self.bf = np.zeros(fc_hidden)
        self.Wp = rng.normal(0.0, math.sqrt(2.0 / fc_hidden), (self.n_cells, fc_hidden))
        self.bp = np.zeros(self.n_cells)
        self.Wv = rng.normal(0.0, math.sqrt(2.0 / fc_hidden), (1, fc_hidden))
        self.bv = np.zeros(1)

    def forward(self, x):
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
        zf = self.Wf @ flat + self.bf
        hf = _relu(zf)
        logits = self.Wp @ hf + self.bp
        value = float(_sigmoid((self.Wv @ hf + self.bv)[0]))
        cache["zf"] = zf
        cache["hf"] = hf
        cache["value"] = value
        return logits, value, cache

    def predict(self, game, temperature=1.0):
        logits, value, _ = self.forward(_board_features(game))
        legal = [_cell_index(r, c) for r, c in game.legal_moves()]
        return _softmax_masked(logits, legal, temperature=temperature), value

    def train_batch(self, xs, policy_targets, value_targets, lr=0.005, value_weight=1.0, l2=1e-5):
        dWc = [np.zeros_like(W) for W in self.Wc]
        dbc = [np.zeros_like(b) for b in self.bc]
        dWf = np.zeros_like(self.Wf)
        dbf = np.zeros_like(self.bf)
        dWp = np.zeros_like(self.Wp)
        dbp = np.zeros_like(self.bp)
        dWv = np.zeros_like(self.Wv)
        dbv = np.zeros_like(self.bv)
        total_loss = 0.0

        for x, policy_target, value_target in zip(xs, policy_targets, value_targets):
            logits, value, cache = self.forward(x)
            legal_mask = policy_target > 0.0
            if not np.any(legal_mask):
                continue

            masked = logits[legal_mask]
            masked = masked - np.max(masked)
            exp_logits = np.exp(np.clip(masked, -60.0, 60.0))
            probs_legal = exp_logits / np.sum(exp_logits)
            probs = np.zeros_like(logits)
            probs[legal_mask] = probs_legal

            value_error = value - value_target
            total_loss += -float(np.sum(policy_target[legal_mask] * np.log(probs_legal + 1e-12)))
            total_loss += 0.5 * value_weight * float(value_error * value_error)

            dlogits = probs - policy_target
            dvalue_logit = value_weight * value_error * value * (1.0 - value)
            hf = cache["hf"]

            dWp += np.outer(dlogits, hf)
            dbp += dlogits
            dWv += dvalue_logit * hf.reshape(1, -1)
            dbv += np.array([dvalue_logit])

            dhf = self.Wp.T @ dlogits + dvalue_logit * self.Wv[0]
            dzf = dhf * _relu_deriv(cache["zf"])
            dWf += np.outer(dzf, cache["flat"])
            dbf += dzf

            dh = self.Wf.T @ dzf
            dh = dh.reshape(cache["flat_shape"])

            for i in range(len(self.Wc) - 1, -1, -1):
                in_shape, cols, z = cache["conv"][i]
                dz = dh * _relu_deriv(z)
                dz_flat = dz.reshape(dz.shape[0], -1)
                W = self.Wc[i]
                C_out, C_in, kh, kw = W.shape
                W_flat = W.reshape(C_out, -1)
                dWc[i] += (dz_flat @ cols.T).reshape(C_out, C_in, kh, kw)
                dbc[i] += dz_flat.sum(axis=1)
                if i > 0:
                    dcols = W_flat.T @ dz_flat
                    dh = _col2im(dcols, C_in, in_shape[1], in_shape[2], kh, kw, 1)

        scale = 1.0 / max(1, xs.shape[0])
        for i in range(len(self.Wc)):
            self.Wc[i] -= lr * (dWc[i] * scale + l2 * self.Wc[i])
            self.bc[i] -= lr * dbc[i] * scale
        self.Wf -= lr * (dWf * scale + l2 * self.Wf)
        self.bf -= lr * dbf * scale
        self.Wp -= lr * (dWp * scale + l2 * self.Wp)
        self.bp -= lr * dbp * scale
        self.Wv -= lr * (dWv * scale + l2 * self.Wv)
        self.bv -= lr * dbv * scale
        return total_loss * scale

    def save(self, path):
        data = {
            "board_size": self.board_size,
            "conv_channels": self.conv_channels,
            "fc_hidden": self.fc_hidden,
            "Wc": self.Wc,
            "bc": self.bc,
            "Wf": self.Wf,
            "bf": self.bf,
            "Wp": self.Wp,
            "bp": self.bp,
            "Wv": self.Wv,
            "bv": self.bv,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        net = cls(
            board_size=data["board_size"],
            conv_channels=tuple(data["conv_channels"]),
            fc_hidden=data["fc_hidden"],
        )
        net.Wc = data["Wc"]
        net.bc = data["bc"]
        net.Wf = data["Wf"]
        net.bf = data["bf"]
        net.Wp = data["Wp"]
        net.bp = data["bp"]
        net.Wv = data["Wv"]
        net.bv = data["bv"]
        return net


def generate_self_play_cnn_examples(net, board_size=7, num_games=24, search_iters=120,
                                    rollouts_per_leaf=1, value_weight=0.1,
                                    temperature_moves=8, out=None):
    from game import Game

    examples = []
    agent = PVMCTSAgent(
        net=net,
        iterations=search_iters,
        rollouts_per_leaf=rollouts_per_leaf,
        value_weight=value_weight,
        rollout_policy="random",
    )
    for game_idx in range(num_games):
        game = Game(size=board_size)
        game_examples = []
        while not game.is_over():
            stats = agent.analyze(game)
            if not stats:
                break
            move_number = len(game.move_history)
            temperature = 1.0 if move_number < temperature_moves else 0.25
            move = _sample_move_from_stats(stats, temperature=temperature)
            if move is None:
                break
            game_examples.append(
                TrainingExample(
                    features=_board_features(game),
                    policy=_policy_from_stats(stats, board_size),
                    player=game.current_player,
                )
            )
            game.make_move(move[0], move[1])

        for ex in game_examples:
            ex.winner = game.winner
            ex.value = 1.0 if game.winner == ex.player else 0.0
        examples.extend(game_examples)

        if out is not None and (game_idx + 1) % 6 == 0:
            print(f"  SP-PV-CNN self-play game {game_idx + 1}/{num_games}", file=out, flush=True)

    return examples


def train_self_play_cnn_policy_value_net(board_size=7, generations=5, games_per_generation=20,
                                         search_iters=100, conv_channels=(24, 24, 16),
                                         fc_hidden=96, epochs=18, batch_size=64,
                                         lr=0.004, replay_size=9000, seed=31, out=None):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    net = PolicyValueCNN(
        board_size=board_size,
        conv_channels=conv_channels,
        fc_hidden=fc_hidden,
        seed=seed,
    )
    replay = deque(maxlen=replay_size)

    for gen in range(generations):
        if out is not None:
            print(
                f"SP-PV-CNN generation {gen + 1}/{generations}: "
                f"{games_per_generation} games, search={search_iters}",
                file=out,
                flush=True,
            )
        examples = generate_self_play_cnn_examples(
            net,
            board_size=board_size,
            num_games=games_per_generation,
            search_iters=search_iters,
            rollouts_per_leaf=1,
            value_weight=min(0.35, 0.08 + 0.06 * gen),
            out=out,
        )
        replay.extend(augment_examples(examples, board_size))
        if not replay:
            raise RuntimeError("No SP-PV-CNN self-play examples were generated")

        train_examples = list(replay)
        xs = np.stack([ex.features for ex in train_examples])
        policies = np.stack([ex.policy for ex in train_examples])
        values = np.array([ex.value for ex in train_examples], dtype=np.float64)
        indices = np.arange(len(train_examples))
        gen_lr = lr * (0.9 ** gen)

        for epoch in range(epochs):
            rng.shuffle(indices)
            losses = []
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                losses.append(
                    net.train_batch(
                        xs[batch],
                        policies[batch],
                        values[batch],
                        lr=gen_lr,
                        value_weight=1.2,
                    )
                )
            if out is not None and ((epoch + 1) % 6 == 0 or epoch == 0):
                print(
                    f"  SP-PV-CNN epoch {epoch + 1}/{epochs}, "
                    f"loss={np.mean(losses):.4f}, replay={len(replay)}",
                    file=out,
                    flush=True,
                )

    return net


def default_self_play_cnn_model_path(board_size):
    return os.path.join(os.path.dirname(__file__), "models", f"sp_pv_cnn_model_s{board_size}.pkl")


def load_or_train_self_play_cnn_pv_mcts(board_size=7, model_path=None, retrain=False,
                                        generations=5, games_per_generation=20,
                                        search_iters=100, conv_channels=(24, 24, 16),
                                        fc_hidden=96, epochs=18, out=None):
    model_path = model_path or default_self_play_cnn_model_path(board_size)
    if not retrain and os.path.exists(model_path):
        if out is not None:
            print(f"Loaded self-play PV-CNN net from {model_path}", file=out)
        return PolicyValueCNN.load(model_path), model_path

    if out is not None:
        print(
            f"Training self-play PV-CNN net: {generations} generations, "
            f"{games_per_generation} games/gen, search={search_iters}",
            file=out,
            flush=True,
        )
    net = train_self_play_cnn_policy_value_net(
        board_size=board_size,
        generations=generations,
        games_per_generation=games_per_generation,
        search_iters=search_iters,
        conv_channels=conv_channels,
        fc_hidden=fc_hidden,
        epochs=epochs,
        out=out,
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    net.save(model_path)
    if out is not None:
        print(f"Saved self-play PV-CNN net to {model_path}", file=out)
    return net, model_path
