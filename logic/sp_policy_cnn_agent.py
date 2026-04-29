"""Self-play CNN policy agent for Game of Y.

This agent is intentionally policy-only: no MCTS, no value head, and no
teacher. It learns from its own sampled games with a REINFORCE-style update
and augments every move with the six symmetries of the triangular board.
At play time it can average policy probabilities across those same symmetries.
"""

import math
import os
import pickle
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

from pv_mcts_agent import _cell_index, _index_to_cell, _num_cells
from sp_pv_cnn_agent import (
    N_CHANNELS,
    SYMMETRIES,
    _board_features,
    _softmax_masked,
    _transform_cell,
    _transform_tensor,
)
from td_cnn_agent import _col2im, _im2col, _relu, _relu_deriv


@dataclass
class PolicyExample:
    features: np.ndarray
    action: int
    player: int
    winner: int = 0
    advantage: float = 0.0


def _transform_action(action, board_size, perm):
    row, col = _index_to_cell(action)
    tr, tc = _transform_cell(row, col, board_size, perm)
    return _cell_index(tr, tc)


def _untransform_policy(policy, board_size, perm):
    out = np.zeros_like(policy)
    for r in range(board_size):
        for c in range(r + 1):
            tr, tc = _transform_cell(r, c, board_size, perm)
            out[_cell_index(r, c)] = policy[_cell_index(tr, tc)]
    return out


def augment_policy_examples(examples, board_size):
    augmented = []
    for ex in examples:
        for perm in SYMMETRIES:
            augmented.append(
                PolicyExample(
                    features=_transform_tensor(ex.features, perm),
                    action=_transform_action(ex.action, board_size, perm),
                    player=ex.player,
                    winner=ex.winner,
                    advantage=ex.advantage,
                )
            )
    return augmented


class PolicyCNN:
    """Small CNN with a single policy head over triangular-board cells."""

    def __init__(self, board_size=7, conv_channels=(32, 32, 24), fc_hidden=128, seed=None):
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
        cache["zf"] = zf
        cache["hf"] = hf
        return logits, cache

    def predict(self, game, temperature=1.0, symmetry_average=True):
        legal = [_cell_index(r, c) for r, c in game.legal_moves()]
        if not legal:
            return np.zeros(self.n_cells, dtype=np.float64)

        x = _board_features(game)
        if not symmetry_average:
            logits, _ = self.forward(x)
            return _softmax_masked(logits, legal, temperature=temperature)

        policy = np.zeros(self.n_cells, dtype=np.float64)
        for perm in SYMMETRIES:
            xt = _transform_tensor(x, perm)
            legal_t = [_transform_action(idx, self.board_size, perm) for idx in legal]
            logits, _ = self.forward(xt)
            pt = _softmax_masked(logits, legal_t, temperature=temperature)
            policy += _untransform_policy(pt, self.board_size, perm)

        total = float(np.sum(policy[legal]))
        if total <= 0.0 or not np.isfinite(total):
            policy[legal] = 1.0 / len(legal)
        else:
            policy /= total
        return policy

    def train_batch(self, xs, actions, advantages, lr=0.004, l2=1e-5, grad_clip=3.0):
        dWc = [np.zeros_like(W) for W in self.Wc]
        dbc = [np.zeros_like(b) for b in self.bc]
        dWf = np.zeros_like(self.Wf)
        dbf = np.zeros_like(self.bf)
        dWp = np.zeros_like(self.Wp)
        dbp = np.zeros_like(self.bp)
        total_loss = 0.0

        for x, action, advantage in zip(xs, actions, advantages):
            logits, cache = self.forward(x)
            legal = [
                _cell_index(r, c)
                for r in range(self.board_size)
                for c in range(r + 1)
                if x[3, r, c] > 0.5
            ]
            if len(legal) == 0:
                continue

            probs = _softmax_masked(logits, legal)
            if probs[action] <= 0.0:
                continue

            weight = float(np.clip(advantage, -1.0, 1.0))
            total_loss += -weight * math.log(float(probs[action]) + 1e-12)

            dlogits = weight * probs
            dlogits[action] -= weight

            hf = cache["hf"]
            dWp += np.outer(dlogits, hf)
            dbp += dlogits

            dhf = self.Wp.T @ dlogits
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
        grads = dWc + dbc + [dWf, dbf, dWp, dbp]
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads))
        clip = min(1.0, grad_clip / (norm * scale + 1e-12))

        for i in range(len(self.Wc)):
            self.Wc[i] -= lr * (clip * dWc[i] * scale + l2 * self.Wc[i])
            self.bc[i] -= lr * clip * dbc[i] * scale
        self.Wf -= lr * (clip * dWf * scale + l2 * self.Wf)
        self.bf -= lr * clip * dbf * scale
        self.Wp -= lr * (clip * dWp * scale + l2 * self.Wp)
        self.bp -= lr * clip * dbp * scale
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
        return net


class SPPolicyCNNAgent:
    """Choose moves directly from a trained self-play policy CNN."""

    def __init__(self, model_path=None, net=None, temperature=0.05, symmetry_average=True):
        if net is None:
            if model_path is None:
                raise ValueError("SPPolicyCNNAgent needs a model_path or a PolicyCNN")
            net = PolicyCNN.load(model_path)
        self.net = net
        self.temperature = temperature
        self.symmetry_average = symmetry_average

    def choose_move(self, game):
        moves = game.legal_moves()
        if not moves:
            return None

        policy = self.net.predict(
            game,
            temperature=self.temperature,
            symmetry_average=self.symmetry_average,
        )
        best_move = None
        best_prob = -1.0
        for move in moves:
            prob = float(policy[_cell_index(move[0], move[1])])
            if prob > best_prob:
                best_prob = prob
                best_move = move
        return best_move if best_move is not None else random.choice(moves)


def _sample_action(policy, legal_moves):
    weights = np.array([policy[_cell_index(r, c)] for r, c in legal_moves], dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return random.choice(legal_moves)
    idx = int(np.random.choice(len(legal_moves), p=weights / total))
    return legal_moves[idx]


def generate_self_play_policy_examples(net, board_size=7, num_games=200,
                                       temperature=1.0, min_temperature=0.35,
                                       out=None):
    from game import Game

    examples = []
    for game_idx in range(num_games):
        game = Game(size=board_size)
        game_examples = []
        while not game.is_over():
            moves = game.legal_moves()
            if not moves:
                break
            progress = len(game.move_history) / max(1, net.n_cells)
            temp = max(min_temperature, temperature * (1.0 - 0.65 * progress))
            policy = net.predict(game, temperature=temp, symmetry_average=True)
            move = _sample_action(policy, moves)
            game_examples.append(
                PolicyExample(
                    features=_board_features(game),
                    action=_cell_index(move[0], move[1]),
                    player=game.current_player,
                )
            )
            game.make_move(move[0], move[1])

        for ex in game_examples:
            ex.winner = game.winner
            ex.advantage = 1.0 if game.winner == ex.player else 0.0
        examples.extend(game_examples)

        if out is not None and (game_idx + 1) % 25 == 0:
            print(f"  SP-Policy-CNN self-play game {game_idx + 1}/{num_games}", file=out, flush=True)

    return examples


def train_self_play_policy_cnn(board_size=7, generations=20, games_per_generation=200,
                               conv_channels=(32, 32, 24), fc_hidden=128,
                               epochs=8, batch_size=96, lr=0.004,
                               replay_size=24000, seed=43, net=None, out=None):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)
    if net is None:
        net = PolicyCNN(
            board_size=board_size,
            conv_channels=conv_channels,
            fc_hidden=fc_hidden,
            seed=seed,
        )
    replay = deque(maxlen=replay_size)

    for gen in range(generations):
        if out is not None:
            print(
                f"SP-Policy-CNN generation {gen + 1}/{generations}: "
                f"{games_per_generation} self-play games",
                file=out,
                flush=True,
            )
        examples = generate_self_play_policy_examples(
            net,
            board_size=board_size,
            num_games=games_per_generation,
            temperature=max(0.55, 1.15 * (0.94 ** gen)),
            min_temperature=0.20,
            out=out,
        )
        replay.extend(augment_policy_examples(examples, board_size))
        if not replay:
            raise RuntimeError("No SP-Policy-CNN self-play examples were generated")

        train_examples = list(replay)
        xs = np.stack([ex.features for ex in train_examples])
        actions = np.array([ex.action for ex in train_examples], dtype=np.int64)
        advantages = np.array([ex.advantage for ex in train_examples], dtype=np.float64)

        indices = np.arange(len(train_examples))
        gen_lr = lr * (0.92 ** gen)
        for epoch in range(epochs):
            rng.shuffle(indices)
            losses = []
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                losses.append(
                    net.train_batch(
                        xs[batch],
                        actions[batch],
                        advantages[batch],
                        lr=gen_lr,
                    )
                )
            if out is not None and (epoch == 0 or (epoch + 1) % 4 == 0):
                print(
                    f"  SP-Policy-CNN epoch {epoch + 1}/{epochs}, "
                    f"loss={np.mean(losses):.4f}, replay={len(replay)}",
                    file=out,
                    flush=True,
                )

    return net


def default_self_play_policy_cnn_model_path(board_size):
    return os.path.join(os.path.dirname(__file__), "models", f"sp_policy_cnn_model_s{board_size}.pkl")


def load_or_train_self_play_policy_cnn(board_size=7, model_path=None, retrain=False,
                                       continue_training=False,
                                       generations=20, games_per_generation=200,
                                       conv_channels=(32, 32, 24), fc_hidden=128,
                                       epochs=8, out=None):
    model_path = model_path or default_self_play_policy_cnn_model_path(board_size)
    if not retrain and not continue_training and os.path.exists(model_path):
        if out is not None:
            print(f"Loaded self-play policy CNN from {model_path}", file=out)
        return PolicyCNN.load(model_path), model_path

    net = None
    if continue_training and os.path.exists(model_path):
        net = PolicyCNN.load(model_path)
        if out is not None:
            print(f"Continuing self-play policy CNN from {model_path}", file=out)

    if out is not None:
        print(
            f"Training self-play policy CNN: {generations} generations, "
            f"{games_per_generation} games/gen, board size={board_size}",
            file=out,
            flush=True,
        )
    net = train_self_play_policy_cnn(
        board_size=board_size,
        generations=generations,
        games_per_generation=games_per_generation,
        conv_channels=conv_channels,
        fc_hidden=fc_hidden,
        epochs=epochs,
        net=net,
        out=out,
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    net.save(model_path)
    if out is not None:
        print(f"Saved self-play policy CNN to {model_path}", file=out)
    return net, model_path
