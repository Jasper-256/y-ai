"""sb3_elo_arena.py — Train SB3 algorithms on Game of Y then ELO rank them.

Supported algorithms
--------------------
Discrete (native, action masking supported):
    maskable_ppo   MaskablePPO from sb3-contrib        ← recommended
    ppo            PPO
    a2c            A2C
    dqn            DQN

Continuous (output = logit vector over cells, masked argmax picks the move):
    ddpg           DDPG
    td3            TD3
    sac            SAC

How the continuous trick works
-------------------------------
DDPG/TD3/SAC normally require a continuous action space.  We give them one:
Box(-1, 1, shape=(n_cells,)).  The actor outputs a score for every cell; the
env masks illegal cells to -1e9, then takes argmax to select the actual move.
The agent never needs to learn that illegal moves exist — the mask handles it.

ELO
---
All trained SB3 models, plus any requested existing agents (Random, Heuristic,
TD(0), TD(λ), TD-CNN, MCTS, PV-MCTS …), compete in a round-robin tournament.
Results feed into a Bradley-Terry solver (same as arena.py) which produces ELO
ratings with standard errors.

Install
-------
    pip install stable-baselines3 sb3-contrib --break-system-packages

Usage
-----
    # Train maskable_ppo + a2c + dqn, then ELO vs Random & Heuristic
    python sb3_elo_arena.py

    # Specific algorithms + more training + self-play
    python sb3_elo_arena.py --algos maskable_ppo td3 sac \\
                            --timesteps 500000 --self-play-stages 2

    # Skip training (load saved models) and just run ELO
    python sb3_elo_arena.py --no-train --algos maskable_ppo td3

    # Add existing trained agents to the ELO field
    python sb3_elo_arena.py --include-td --include-mcts --mcts-iters 500

    # Save ELO report to logic/output/elo_report.txt
    python sb3_elo_arena.py --output elo_report.txt

    # Full self-play experiment (retrain even if saved model exists)
    python sb3_elo_arena.py --algos maskable_ppo ppo a2c dqn \\
                            --retrain --timesteps 1000000 --self-play-stages 3 \\
                            --include-td --games-per-matchup 100
"""

import os
import sys
import copy
import argparse
import subprocess

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Make logic/ importable when this file lives alongside the other modules
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)


# ═══════════════════════════════════════════════════════════════════════════
# COORDINATE HELPERS  (kept self-contained so the file works standalone)
# ═══════════════════════════════════════════════════════════════════════════

def _cell_index(row: int, col: int) -> int:
    return row * (row + 1) // 2 + col


def _cell_from_index(idx: int):
    row = 0
    while idx >= row + 1:
        idx -= row + 1
        row += 1
    return row, idx


def _n_cells(board_size: int) -> int:
    return board_size * (board_size + 1) // 2


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — SB3 AGENT WRAPPER
#
# Converts any trained SB3 model into the choose_move(game) interface that
# arena.py expects.  Three inference paths are handled:
#
#   is_maskable  — MaskablePPO: pass action_masks kwarg to model.predict()
#   is_continuous— DDPG/TD3/SAC: output is a float vector, mask + argmax
#   default      — PPO/A2C/DQN: scalar action; fall back to random if illegal
# ═══════════════════════════════════════════════════════════════════════════

class SB3AgentWrapper:
    """Wrap a trained SB3 model so it can be used inside arena.py tournaments.

    Parameters
    ----------
    model :
        Any trained stable-baselines3 / sb3-contrib model.
    board_size : int
    is_maskable : bool
        Set True for MaskablePPO — enables the action_masks kwarg.
    is_continuous : bool
        Set True for DDPG/TD3/SAC — output is a float vector of length n_cells;
        illegal cells are set to -1e9 before argmax.
    deterministic : bool
        True (default) for greedy tournament play.
    """

    def __init__(self, model, board_size: int = 7,
                 is_maskable: bool = False,
                 is_continuous: bool = False,
                 deterministic: bool = True):
        self.model = model
        self.board_size = board_size
        self.n_cells = _n_cells(board_size)
        self.is_maskable = is_maskable
        self.is_continuous = is_continuous
        self.deterministic = deterministic

    # ------------------------------------------------------------------
    # Main interface used by arena.py
    # ------------------------------------------------------------------

    def choose_move(self, game):
        """Return (row, col) — the model's best legal move."""
        moves = game.legal_moves()
        if not moves:
            return None

        # Grab an immediate win if available — same shortcut all TD agents use
        me = game.current_player
        for move in moves:
            child = game.copy()
            child.make_move(move[0], move[1])
            if child.winner == me:
                return move

        obs  = self._obs(game)
        mask = self._mask(game)

        if self.is_maskable:
            # MaskablePPO expects action_masks as a keyword argument
            action, _ = self.model.predict(
                obs, action_masks=mask, deterministic=self.deterministic
            )
            action = int(action)

        elif self.is_continuous:
            # DDPG / TD3 / SAC output a float vector of length n_cells.
            # Zero-out illegal cells with a very negative score, then argmax.
            raw, _ = self.model.predict(obs, deterministic=self.deterministic)
            raw = np.array(raw, dtype=np.float64)
            raw[~mask] = -1e9
            action = int(np.argmax(raw))

        else:
            # Scalar-action policy (PPO, A2C, DQN).
            # If the model selects an illegal cell, fall back to a random legal one.
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            action = int(action)
            if not mask[action]:
                action = int(np.random.choice(np.where(mask)[0]))

        row, col = _cell_from_index(action)
        return row, col

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _obs(self, game) -> np.ndarray:
        """Encode board from game.current_player's perspective (matches training)."""
        board = game.board
        me    = game.current_player     # always encode from whoever's turn it is
        opp   = 3 - me
        feats = []
        for r in range(self.board_size):
            for c in range(r + 1):
                v = board.get_cell(r, c)
                feats.append(1.0 if v == 0  else 0.0)
                feats.append(1.0 if v == me  else 0.0)
                feats.append(1.0 if v == opp else 0.0)
        feats.append(1.0)           # bias
        return np.array(feats, dtype=np.float32)

    def _mask(self, game) -> np.ndarray:
        mask = np.zeros(self.n_cells, dtype=bool)
        for r, c in game.legal_moves():
            mask[_cell_index(r, c)] = True
        return mask


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — GYMNASIUM ENVIRONMENTS
#
# Two variants:
#   YGameEnv          Discrete(n_cells)   — PPO, A2C, DQN, MaskablePPO
#   ContinuousYGameEnv Box(n_cells,)      — DDPG, TD3, SAC
#
# Both share the same observation space and reward structure.
# The opponent (any agent with choose_move()) is called automatically inside
# step(), so the RL agent only ever sees states where it is *its* turn.
# ═══════════════════════════════════════════════════════════════════════════

class YGameEnv(gym.Env):
    """Discrete action-space Gymnasium environment for Game of Y.

    Observation : float32 vector of length n_cells * 3 + 1
                  (empty / mine / opponent one-hot per cell + bias)
    Action      : int in [0, n_cells)   — flat cell index
    Reward      : +1 win | -1 loss or illegal move | 0 mid-game
    info keys   : "action_mask" (bool ndarray, True = legal)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, board_size: int = 7, opponent=None,
                 agent_player: int = 1, render_mode=None,
                 illegal_move_mode: str = "penalise"):
        super().__init__()
        from game import Game           # validate Cython is compiled
        self.board_size      = board_size
        self.n_cells         = _n_cells(board_size)
        self.agent_player    = agent_player
        self.opponent_player = 3 - agent_player
        self.render_mode     = render_mode
        self.illegal_mode    = illegal_move_mode
        self._opponent       = opponent     # can be swapped between episodes
        self._game           = None
        self._active_opp     = None         # resolved opponent for current ep.

        obs_dim = self.n_cells * 3 + 1
        self.observation_space = spaces.Box(0.0, 1.0, (obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Discrete(self.n_cells)

    # Allow swapping opponent mid-run (used for self-play curriculum)
    @property
    def opponent(self):
        return self._opponent

    @opponent.setter
    def opponent(self, opp):
        self._opponent = opp

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        from game import Game
        self._game = Game(size=self.board_size)
        self._active_opp = self._resolve_opponent()
        # If opponent moves first, let them go now
        if self._game.current_player == self.opponent_player:
            self._play_opponent()
        obs  = self._get_obs()
        info = {"action_mask": self._get_mask()}
        return obs, info

    def step(self, action: int):
        assert self._game is not None, "Call reset() first."
        row, col = _cell_from_index(int(action))

        # ── Illegal move handling ──────────────────────────────────────
        if self._game.board.get_cell(row, col) != 0:
            if self.illegal_mode == "raise":
                raise ValueError(f"Illegal action {action}: ({row},{col}) occupied.")
            return (self._get_obs(), -1.0, True, False,
                    {"action_mask": self._get_mask(), "illegal_move": True})

        # ── Agent plays ────────────────────────────────────────────────
        self._game.make_move(row, col)
        if self._game.is_over():
            return self._terminal()

        # ── Opponent plays ─────────────────────────────────────────────
        self._play_opponent()
        if self._game.is_over():
            return self._terminal()

        return self._get_obs(), 0.0, False, False, {"action_mask": self._get_mask()}

    # Called by ActionMasker wrapper (sb3-contrib)
    def action_masks(self) -> np.ndarray:
        return self._get_mask()

    def render(self):
        s = self._render_ansi()
        if self.render_mode == "human":
            print(s)
        elif self.render_mode == "ansi":
            return s

    def close(self):
        self._game = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_opponent(self):
        if self._opponent is not None:
            return self._opponent
        from random_agent import RandomAgent
        return RandomAgent()

    def _get_obs(self) -> np.ndarray:
        board = self._game.board
        me    = self.agent_player
        opp   = self.opponent_player
        feats = []
        for r in range(self.board_size):
            for c in range(r + 1):
                v = board.get_cell(r, c)
                feats.append(1.0 if v == 0  else 0.0)
                feats.append(1.0 if v == me  else 0.0)
                feats.append(1.0 if v == opp else 0.0)
        feats.append(1.0)
        return np.array(feats, dtype=np.float32)

    def _get_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_cells, dtype=bool)
        if not self._game.is_over():
            for r, c in self._game.legal_moves():
                mask[_cell_index(r, c)] = True
        return mask

    def _play_opponent(self):
        if self._game.is_over():
            return
        move = self._active_opp.choose_move(self._game)
        if move is not None:
            self._game.make_move(move[0], move[1])

    def _terminal(self):
        r = 1.0 if self._game.winner == self.agent_player else -1.0
        return (self._get_obs(), r, True, False,
                {"action_mask": np.zeros(self.n_cells, dtype=bool)})

    def _render_ansi(self) -> str:
        syms  = {0: ".", 1: "R", 2: "B"}
        board = self._game.board
        lines = []
        for r in range(self.board_size):
            pad = " " * (self.board_size - r - 1)
            lines.append(pad + " ".join(syms[board.get_cell(r, c)] for c in range(r+1)))
        turn = "R" if self._game.current_player == 1 else "B"
        lines.append(f"Turn: {turn}   Moves: {len(self._game.move_history)}")
        if self._game.winner:
            w = "R" if self._game.winner == 1 else "B"
            lines.append(f"*** {w} wins! ***")
        return "\n".join(lines)


class ContinuousYGameEnv(YGameEnv):
    """Continuous action-space variant for DDPG, TD3, SAC.

    Action space: Box(-1, 1, shape=(n_cells,))
    The env applies the legal mask (sets illegal entries to -1e9) and picks
    the argmax — the agent never needs to know about illegal moves.
    All other behaviour is identical to YGameEnv.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_cells,), dtype=np.float32
        )

    def step(self, action):
        assert self._game is not None, "Call reset() first."
        mask = self._get_mask()
        if not np.any(mask):
            return self._get_obs(), 0.0, True, False, {"action_mask": mask}

        # Mask illegal, argmax → cell index
        raw = np.array(action, dtype=np.float64)
        raw[~mask] = -1e9
        cell_idx = int(np.argmax(raw))
        row, col = _cell_from_index(cell_idx)

        self._game.make_move(row, col)
        if self._game.is_over():
            return self._terminal()

        self._play_opponent()
        if self._game.is_over():
            return self._terminal()

        return self._get_obs(), 0.0, False, False, {"action_mask": self._get_mask()}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — ALGORITHM REGISTRY & HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

DISCRETE_ALGOS   = {"maskable_ppo", "ppo", "a2c", "dqn"}
CONTINUOUS_ALGOS = {"ddpg", "td3", "sac"}
ALL_ALGOS        = DISCRETE_ALGOS | CONTINUOUS_ALGOS


def _import_algo(name: str):
    """Return the SB3 algorithm class for a given name string."""
    name = name.lower()
    if name == "maskable_ppo":
        from sb3_contrib import MaskablePPO
        return MaskablePPO
    if name == "ppo":
        from stable_baselines3 import PPO
        return PPO
    if name == "a2c":
        from stable_baselines3 import A2C
        return A2C
    if name == "dqn":
        from stable_baselines3 import DQN
        return DQN
    if name == "ddpg":
        from stable_baselines3 import DDPG
        return DDPG
    if name == "td3":
        from stable_baselines3 import TD3
        return TD3
    if name == "sac":
        from stable_baselines3 import SAC
        return SAC
    raise ValueError(f"Unknown algorithm '{name}'. "
                     f"Valid choices: {sorted(ALL_ALGOS)}")


def _default_hparams(name: str) -> dict:
    """Sensible default hyperparameters for each algorithm."""
    base = {"verbose": 0, "device": "auto"}
    if name == "maskable_ppo":
        return {"policy": "MlpPolicy", "n_steps": 2048, "batch_size": 64,
                "n_epochs": 10, "learning_rate": 3e-4, "ent_coef": 0.01, **base}
    if name == "ppo":
        return {"policy": "MlpPolicy", "n_steps": 2048, "batch_size": 64,
                "n_epochs": 10, "learning_rate": 3e-4, "ent_coef": 0.01, **base}
    if name == "a2c":
        return {"policy": "MlpPolicy", "n_steps": 5,
                "learning_rate": 7e-4, "ent_coef": 0.01, **base}
    if name == "dqn":
        return {"policy": "MlpPolicy", "learning_rate": 1e-4,
                "buffer_size": 100_000, "batch_size": 32,
                "exploration_fraction": 0.25,
                "exploration_final_eps": 0.05, **base}
    if name in ("ddpg", "td3"):
        return {"policy": "MlpPolicy", "learning_rate": 1e-4,
                "buffer_size": 100_000, "batch_size": 100, **base}
    if name == "sac":
        return {"policy": "MlpPolicy", "learning_rate": 3e-4,
                "buffer_size": 100_000, "batch_size": 256,
                "ent_coef": "auto", **base}
    return {"policy": "MlpPolicy", **base}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_agent(
    algo_name: str,
    board_size: int = 7,
    timesteps: int = 300_000,
    opponent=None,
    save_dir: str = "models/sb3",
    self_play_stages: int = 1,
    retrain: bool = False,
    out=None,
) -> tuple:
    """Train one SB3 algorithm and return (SB3AgentWrapper, model_path).

    Training curriculum
    -------------------
    Stage 0 : Train against `opponent` (default RandomAgent) for `timesteps`.
    Stage 1…N: Freeze current model as the new opponent, train another
               `timesteps` steps (self-play).  Controlled by `self_play_stages`.

    If a saved model exists and `retrain=False`, training is skipped and the
    saved model is loaded directly.

    Parameters
    ----------
    algo_name : str
    board_size : int
    timesteps : int
        Steps per stage.
    opponent : agent or None
        First-stage opponent.  None → RandomAgent.
    save_dir : str
    self_play_stages : int
        Number of self-play stages after the initial stage.
    retrain : bool
        Force retraining even if a saved model exists.
    out : file-like
    """
    if out is None:
        out = sys.stdout

    algo_name  = algo_name.lower()
    is_discrete   = algo_name in DISCRETE_ALGOS
    is_maskable   = algo_name == "maskable_ppo"
    is_continuous = algo_name in CONTINUOUS_ALGOS

    AlgoClass = _import_algo(algo_name)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{algo_name}_s{board_size}.zip")

    # ── Load existing model unless retrain requested ─────────────────────
    if not retrain and os.path.exists(model_path):
        print(f"[{algo_name}] Loading saved model from {model_path}", file=out, flush=True)
        model = AlgoClass.load(model_path)
        return (
            SB3AgentWrapper(model, board_size=board_size,
                            is_maskable=is_maskable, is_continuous=is_continuous),
            model_path,
        )

    # ── Build env factory ─────────────────────────────────────────────────
    EnvClass = ContinuousYGameEnv if is_continuous else YGameEnv

    def make_env(opp):
        env = EnvClass(board_size=board_size, opponent=opp)
        if is_maskable:
            from sb3_contrib.common.wrappers import ActionMasker
            env = ActionMasker(env, lambda e: e.action_masks())
        return env

    if opponent is None:
        from random_agent import RandomAgent
        opponent = RandomAgent()

    hparams = _default_hparams(algo_name)

    # ── Stage 0: train against initial opponent ───────────────────────────
    print(f"\n[{algo_name}] Stage 0 — training {timesteps:,} steps vs {type(opponent).__name__}",
          file=out, flush=True)
    env   = make_env(opponent)
    model = AlgoClass(env=env, **hparams)
    model.learn(total_timesteps=timesteps, progress_bar=False)

    # ── Self-play stages ─────────────────────────────────────────────────
    for stage in range(1, self_play_stages + 1):
        # Deep-copy current model, wrap it, use as frozen opponent
        import tempfile as _tf
        _tmp = _tf.mktemp(suffix=".zip")
        model.save(_tmp)
        _frozen_model = AlgoClass.load(_tmp)
        import os as _os
        for _p in [_tmp, _tmp + ".zip"]:
            try: _os.unlink(_p)
            except FileNotFoundError: pass
        frozen = SB3AgentWrapper(
            _frozen_model, board_size=board_size,
            is_maskable=is_maskable, is_continuous=is_continuous,
            deterministic=True,
        )
        print(f"[{algo_name}] Stage {stage}/{self_play_stages} — self-play {timesteps:,} steps",
              file=out, flush=True)
        sp_env = make_env(frozen)
        model.set_env(sp_env)
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=False)

    model.save(model_path)
    print(f"[{algo_name}] Saved to {model_path}", file=out, flush=True)

    return (
        SB3AgentWrapper(model, board_size=board_size,
                        is_maskable=is_maskable, is_continuous=is_continuous),
        model_path,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — AGENT ROSTER
#
# Collects all agents (trained SB3 models + requested existing agents) into
# a dict {label: agent} ready for the tournament.
# ═══════════════════════════════════════════════════════════════════════════

def build_roster(args, out=None) -> dict:
    """Return {label: agent} for the ELO tournament."""
    if out is None:
        out = sys.stdout

    from random_agent import RandomAgent
    from heuristic_agent import HeuristicAgent

    agents = {}

    # ── SB3 agents ────────────────────────────────────────────────────────
    for algo in args.algos:
        algo = algo.lower()
        if algo not in ALL_ALGOS:
            print(f"Warning: unknown algorithm '{algo}', skipping.", file=out)
            continue

        if args.no_train:
            model_path = os.path.join(args.save_dir, f"{algo}_s{args.board_size}.zip")
            if not os.path.exists(model_path):
                print(f"Warning: no saved model at {model_path}, skipping {algo}.", file=out)
                continue
            AlgoClass = _import_algo(algo)
            model     = AlgoClass.load(model_path)
            wrapper   = SB3AgentWrapper(
                model, board_size=args.board_size,
                is_maskable=(algo == "maskable_ppo"),
                is_continuous=(algo in CONTINUOUS_ALGOS),
            )
        else:
            wrapper, _ = train_agent(
                algo,
                board_size=args.board_size,
                timesteps=args.timesteps,
                save_dir=args.save_dir,
                self_play_stages=args.self_play_stages,
                retrain=args.retrain,
                out=out,
            )
        agents[algo.upper()] = wrapper

    # ── Always-present baselines ─────────────────────────────────────────
    agents["Random"]    = RandomAgent()
    agents["Heuristic"] = HeuristicAgent()
    print("Loaded baselines: Random, Heuristic", file=out, flush=True)

    # ── Optional: existing TD agents ─────────────────────────────────────
    if args.include_td:
        from td_agent import TDAgent
        from td_lambda_agent import TDLambdaAgent
        from td_cnn_agent import TDCNNAgent
        for cls, tag, path in [
            (TDAgent,       "TD(0)",  f"models/td_model_s{args.board_size}.pkl"),
            (TDLambdaAgent, "TD(λ)",  f"models/td_lambda_model_s{args.board_size}.pkl"),
            (TDCNNAgent,    "TD-CNN", f"models/td_cnn_model_s{args.board_size}.pkl"),
        ]:
            full = os.path.join(_this_dir, path)
            if os.path.exists(full):
                agents[tag] = cls.load(full)
                print(f"Loaded {tag} from {full}", file=out)
            else:
                print(f"Warning: {tag} model not found at {full}, skipping.", file=out)

    # ── Optional: MCTS ───────────────────────────────────────────────────
    if args.include_mcts:
        from mcts import MCTSAgent
        label = f"MCTS({args.mcts_iters})"
        agents[label] = MCTSAgent(iterations=args.mcts_iters)
        print(f"Loaded {label}", file=out, flush=True)

    # ── Optional: PV-MCTS (from pv_mcts_agent.py) ────────────────────────
    if args.include_pv_mcts:
        from pv_mcts_agent import load_or_train_pv_mcts, PVMCTSAgent
        net, path = load_or_train_pv_mcts(board_size=args.board_size, out=out)
        agents[f"PV-MCTS({args.mcts_iters})"] = PVMCTSAgent(net=net, iterations=args.mcts_iters)
        print(f"Loaded PV-MCTS from {path}", file=out, flush=True)

    return agents


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — ELO TOURNAMENT
#
# Delegates to arena.py for the actual round-robin game running and for the
# Bradley-Terry ELO solver.  We only add a pretty leaderboard printer.
# ═══════════════════════════════════════════════════════════════════════════

def run_elo_tournament(agents: dict, games_per_matchup: int = 50,
                       board_size: int = 7, out=None):
    """Run round-robin tournament and return (results, elo_ratings).

    Returns
    -------
    results : dict  {(name1, name2): {1: wins_for_name1, 2: wins_for_name2}}
    elo     : list of (name, elo_rating, standard_error) sorted best → worst
    """
    if out is None:
        out = sys.stdout

    # Re-use arena.py's battle-tested infrastructure
    from arena import run_tournament, print_summary, fit_bradley_terry_elo

    print(f"\n{'='*65}", file=out)
    print(f"ELO TOURNAMENT", file=out)
    print(f"  Agents  : {list(agents.keys())}", file=out)
    print(f"  Board   : size {board_size}", file=out)
    print(f"  Games   : {games_per_matchup} per ordered pair "
          f"({games_per_matchup * 2} per unordered pair)", file=out)
    print(f"{'='*65}", file=out)

    results = run_tournament(agents, games_per_matchup=games_per_matchup,
                             board_size=board_size, out=out)
    print_summary(agents, results, games_per_matchup, out=out)

    elo = fit_bradley_terry_elo(list(agents.keys()), results)
    return results, elo


def print_elo_table(elo_ratings, out=None):
    """Pretty-print the final ELO leaderboard."""
    if out is None:
        out = sys.stdout
    print(f"\n{'='*55}", file=out)
    print("FINAL ELO LEADERBOARD  (Bradley-Terry)", file=out)
    print(f"{'='*55}", file=out)
    print(f"{'Rank':<5}  {'Agent':<24}  {'ELO':>6}  {'±SE':>6}", file=out)
    print(f"{'-'*55}", file=out)
    ranked = sorted(elo_ratings, key=lambda x: -x[1])
    for rank, (name, rating, se) in enumerate(ranked, 1):
        bar = "█" * max(1, int((rating - 1200) / 40))   # mini bar for quick scan
        print(f"  {rank:<4} {name:<24} {int(round(rating)):>6}  ±{int(round(se)):<5}  {bar}",
              file=out)
    print(f"{'='*55}", file=out)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Train SB3 algorithms on Game of Y and rank them via ELO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Which algorithms ──────────────────────────────────────────────────
    p.add_argument(
        "--algos", nargs="+",
        default=["maskable_ppo", "a2c", "dqn"],
        help=f"Algorithms to train/evaluate. Choices: {sorted(ALL_ALGOS)}",
    )

    # ── Training settings ─────────────────────────────────────────────────
    p.add_argument("--board-size",        type=int,   default=7)
    p.add_argument("--timesteps",         type=int,   default=300_000,
                   help="Timesteps per training stage.")
    p.add_argument("--self-play-stages",  type=int,   default=1,
                   help="Self-play stages after initial training (0 = none).")
    p.add_argument("--retrain",           action="store_true",
                   help="Force retraining even if a saved model exists.")
    p.add_argument("--no-train",          action="store_true",
                   help="Skip training; load saved models and run ELO only.")
    p.add_argument("--save-dir",          type=str,   default="models/sb3",
                   help="Directory for saved SB3 model files.")

    # ── Tournament settings ───────────────────────────────────────────────
    p.add_argument("--games-per-matchup", type=int,   default=50,
                   help="Games per ordered pair in the ELO tournament.")

    # ── Extra agents ──────────────────────────────────────────────────────
    p.add_argument("--include-td",        action="store_true",
                   help="Add TD(0), TD(λ), TD-CNN to the tournament.")
    p.add_argument("--include-mcts",      action="store_true",
                   help="Add plain MCTS to the tournament.")
    p.add_argument("--include-pv-mcts",   action="store_true",
                   help="Add PV-MCTS to the tournament.")
    p.add_argument("--mcts-iters",        type=int,   default=500)

    # ── Output ───────────────────────────────────────────────────────────
    p.add_argument("--output", type=str, default=None,
                   help="Also write output to logic/output/<this file>.")

    return p.parse_args()


def main():
    args = _parse_args()

    # ── Compile Cython ────────────────────────────────────────────────────
    print("Compiling Cython modules…")
    subprocess.check_call(
        [sys.executable, "setup_cython.py", "build_ext", "--inplace"],
        cwd=_this_dir,
    )
    print("Cython modules ready.\n")

    # ── Output tee ───────────────────────────────────────────────────────
    from tee import Tee
    out = sys.stdout
    out_f = None
    if args.output:
        output_dir = os.path.join(_this_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        out_f = open(os.path.join(output_dir, args.output), "w", encoding="utf-8")
        out = Tee(sys.stdout, out_f)

    try:
        agents = build_roster(args, out=out)

        if len(agents) < 2:
            print("Need at least 2 agents to run a tournament.", file=out)
            return

        results, elo = run_elo_tournament(
            agents,
            games_per_matchup=args.games_per_matchup,
            board_size=args.board_size,
            out=out,
        )
        print_elo_table(elo, out=out)

    finally:
        if out_f:
            out_f.close()


if __name__ == "__main__":
    main()