"""Gymnasium environment for Game of Y.

Usage
-----
    from y_game_env import YGameEnv

    env = YGameEnv(board_size=7)                   # random opponent
    env = YGameEnv(board_size=7, opponent=MCTSAgent(iterations=200))

    obs, info = env.reset()
    action_mask = info["action_mask"]              # bool array, True = legal

    obs, reward, terminated, truncated, info = env.step(action)

Action masking with MaskablePPO (sb3-contrib)
----------------------------------------------
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    def mask_fn(env):
        return env.action_masks()

    env = ActionMasker(YGameEnv(), mask_fn)
    model = MaskablePPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500_000)

Notes
-----
* Cython modules (board, game, mcts) must be compiled before importing.
  Run:  python setup_cython.py build_ext --inplace
* The agent always plays as `agent_player` (default 1 = Red).
  The opponent handles the other side immediately after each agent step.
* Reward:  +1 win  |  -1 loss or illegal move  |  0 otherwise
* The game never truncates — Y always terminates in finite moves.
"""

import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Ensure logic/ is importable when this file lives alongside the other modules.
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell_index(row: int, col: int) -> int:
    """Flat index for cell (row, col): matches pv_mcts_agent._cell_index."""
    return row * (row + 1) // 2 + col


def _cell_from_index(index: int):
    """Inverse of _cell_index → (row, col)."""
    row = 0
    while index >= row + 1:
        index -= row + 1
        row += 1
    return row, index


def _n_cells(board_size: int) -> int:
    return board_size * (board_size + 1) // 2


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class YGameEnv(gym.Env):
    """Single-agent Gymnasium wrapper for Game of Y.

    Parameters
    ----------
    board_size : int
        Side length of the triangular board (default 7).
    opponent : agent or None
        Any object with a ``choose_move(game) -> (row, col)`` method.
        Defaults to ``RandomAgent``.
    agent_player : int (1 or 2)
        Which player the RL agent controls. The opponent plays the other.
    render_mode : str or None
        ``"human"`` prints to stdout; ``"ansi"`` returns a string.
    illegal_move_mode : str
        ``"penalise"`` (default) — return reward -1 and terminate.
        ``"raise"``  — raise ValueError on illegal action.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        board_size: int = 7,
        opponent=None,
        agent_player: int = 1,
        render_mode: str | None = None,
        illegal_move_mode: str = "penalise",
    ):
        super().__init__()

        from game import Game  # noqa: F401 — validates Cython is compiled

        self.board_size = board_size
        self.n_cells = _n_cells(board_size)
        self.agent_player = agent_player
        self.opponent_player = 3 - agent_player
        self.render_mode = render_mode
        self.illegal_move_mode = illegal_move_mode

        if opponent is None:
            from random_agent import RandomAgent
            self.opponent = RandomAgent()
        else:
            self.opponent = opponent

        # ── Spaces ──────────────────────────────────────────────────────
        # Observation: for each of the n_cells cells, 3 one-hot features
        #   [empty, mine, opponent] + 1 bias = n_cells * 3 + 1 floats.
        # This mirrors the TDAgent / TDLambdaAgent encoding exactly.
        obs_dim = self.n_cells * 3 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action: integer index in [0, n_cells)
        self.action_space = spaces.Discrete(self.n_cells)

        self._game = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        from game import Game
        self._game = Game(size=self.board_size)

        # If the opponent moves first, execute their first move now so the
        # agent always receives a state where it is their turn.
        if self._game.current_player == self.opponent_player:
            self._play_opponent()

        obs = self._get_obs()
        info = {"action_mask": self._get_action_mask()}
        return obs, info

    def step(self, action: int):
        assert self._game is not None, "Call reset() before step()."
        assert not self._game.is_over(), "Game is already over; call reset()."

        row, col = _cell_from_index(int(action))

        # ── Illegal move ────────────────────────────────────────────────
        if self._game.board.get_cell(row, col) != 0:
            if self.illegal_move_mode == "raise":
                raise ValueError(
                    f"Illegal action {action}: cell ({row},{col}) is occupied."
                )
            # "penalise" mode
            obs = self._get_obs()
            info = {
                "action_mask": self._get_action_mask(),
                "illegal_move": True,
            }
            return obs, -1.0, True, False, info

        # ── Agent's move ─────────────────────────────────────────────────
        self._game.make_move(row, col)

        if self._game.is_over():
            return self._terminal_step()

        # ── Opponent's move ───────────────────────────────────────────────
        self._play_opponent()

        if self._game.is_over():
            return self._terminal_step()

        obs = self._get_obs()
        info = {"action_mask": self._get_action_mask()}
        return obs, 0.0, False, False, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "human":
            print(self._render_ansi())

    def close(self):
        self._game = None

    # ------------------------------------------------------------------
    # Action masking helper (for sb3-contrib MaskablePPO)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask of legal actions (True = legal).

        Call this via an ActionMasker wrapper or directly from your
        training loop.
        """
        return self._get_action_mask()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Encode the board from the agent's perspective."""
        board = self._game.board
        me = self.agent_player
        opp = self.opponent_player
        feats = []
        for r in range(self.board_size):
            for c in range(r + 1):
                v = board.get_cell(r, c)
                feats.append(1.0 if v == 0 else 0.0)   # empty
                feats.append(1.0 if v == me else 0.0)   # mine
                feats.append(1.0 if v == opp else 0.0)  # opponent
        feats.append(1.0)  # bias
        return np.array(feats, dtype=np.float32)

    def _get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_cells, dtype=bool)
        if not self._game.is_over():
            for r, c in self._game.legal_moves():
                mask[_cell_index(r, c)] = True
        return mask

    def _play_opponent(self):
        """Let the opponent make one move (no-op if game is over)."""
        if self._game.is_over():
            return
        move = self.opponent.choose_move(self._game)
        if move is not None:
            self._game.make_move(move[0], move[1])

    def _terminal_step(self):
        reward = 1.0 if self._game.winner == self.agent_player else -1.0
        obs = self._get_obs()
        info = {"action_mask": np.zeros(self.n_cells, dtype=bool)}
        return obs, reward, True, False, info

    def _render_ansi(self) -> str:
        symbols = {0: ".", 1: "R", 2: "B"}
        board = self._game.board
        lines = []
        for r in range(self.board_size):
            indent = " " * (self.board_size - r - 1)
            row_str = " ".join(symbols[board.get_cell(r, c)] for c in range(r + 1))
            lines.append(indent + row_str)
        turn_sym = "R" if self._game.current_player == 1 else "B"
        lines.append(f"Turn: {turn_sym}   Moves played: {len(self._game.move_history)}")
        if self._game.winner:
            winner_sym = "R" if self._game.winner == 1 else "B"
            lines.append(f"*** {winner_sym} wins! ***")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gymnasium registration (optional — allows gym.make("YGame-v0"))
# ---------------------------------------------------------------------------

def register_env():
    """Call once at import time or from your training script."""
    from gymnasium.envs.registration import register
    register(
        id="YGame-v0",
        entry_point="y_game_env:YGameEnv",
        kwargs={"board_size": 7},
    )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess, sys as _sys
    # Compile Cython if needed
    _logic_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.check_call(
        [_sys.executable, "setup_cython.py", "build_ext", "--inplace"],
        cwd=_logic_dir,
    )

    env = YGameEnv(board_size=5, render_mode="human")
    obs, info = env.reset()

    print(f"obs shape : {obs.shape}")
    print(f"action space : {env.action_space}")
    print(f"legal actions: {info['action_mask'].sum()} / {env.n_cells}\n")

    done = False
    total_reward = 0.0
    steps = 0
    while not done:
        legal = np.where(info["action_mask"])[0]
        action = int(np.random.choice(legal))        # random legal agent
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    env.render()
    print(f"\nEpisode finished in {steps} steps  |  total reward: {total_reward}")
    env.close()