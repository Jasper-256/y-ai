#!/usr/bin/env python3
"""Training utilities and standalone training runner.

This module serves two purposes:

Shared training loop
- `train()` is the unified TD training loop used by every TD agent
- If checkpoint evaluations are enabled, `train()` also uses
  `_copy_agent_for_checkpoint()` and `_run_checkpoint_evaluations()`

Standalone `training.py` process
- Train agents directly from this module
- Checkpoint evaluations are intended to be used here rather than in `arena.py`

Usage:
    python training.py                                   # default usage
    python training.py --board-size 9 --hidden-size 256  # custom settings
    python training.py --save models/td_model_s7         # specify save path (.pkl optional)
    python training.py --output td_out.txt               # tee output to file

Note: checkpoint evaluations (when enabled) run three mini-tournaments:
- current vs random agent
- current vs heuristic agent (current vs previous is handled in `_run_checkpoint_evaluations()`)
- current vs the previous checkpoint snapshot
"""

# ==============================================================================
# Shared training loop helpers (used in `arena.py` training flow)
# ==============================================================================

import sys
import copy
import random

# Progress log interval and default spacing when checkpoints=True
_CHECKPOINT_EVERY_N_GAMES = 100


def _copy_agent_for_checkpoint(model):
    """Deep-copy the agent for evaluation; clears TD caches and disables training."""
    snap = copy.deepcopy(model)
    snap.training = False
    snap.reset_episode()
    return snap


def _run_checkpoint_evaluations(model, previous_agent, board_size, iteration, out=sys.stdout):
    """Run three mini-tournaments: current vs random, heuristic, and previous-checkpoint agent."""
    from arena import run_tournament
    from heuristic_agent import HeuristicAgent
    from random_agent import RandomAgent

    was_training = model.training
    model.training = False
    try:
        bs = board_size or model.board_size
        rnd = RandomAgent()
        heur = HeuristicAgent()
        games_per_matchup = 50

        print(f"\n--- Checkpoint evaluation after game {iteration} ---", file=out, flush=True)

        matchups = [
            ("current vs random", {"Current": model, "Random": rnd}),
            ("current vs heuristic", {"Current": model, "Heuristic": heur}),
            ("current vs previous", {"Current": model, "Previous": previous_agent}),
        ]
        for label, agents in matchups:
            print(f"\n{label}", file=out, flush=True)
            if label == "current vs previous":
                run_tournament(agents, games_per_matchup=1, board_size=bs, out=out)
            else:
                run_tournament(agents, games_per_matchup=games_per_matchup, board_size=bs, out=out)
    finally:
        model.training = was_training


def train(model, num_games=1000, opponent=None, board_size=None, checkpoints=None, out=sys.stdout):
    """Unified training loop for all TD agents.

    `checkpoints` may be:
    - None or False: no checkpoint evaluations.
    - True: evaluate after every 100th game (100, 200, … up to num_games).
    - An iterable of 1-based game indices (e.g. [50, 100, 150]).
    """
    from game import Game

    bs = board_size or model.board_size
    model.training = True
    self_play = opponent is None

    if checkpoints is True:
        checkpoint_set = set(
            range(_CHECKPOINT_EVERY_N_GAMES, num_games + 1, _CHECKPOINT_EVERY_N_GAMES)
        )
        if not checkpoint_set:
            checkpoint_set = None
    elif checkpoints:
        checkpoint_set = set(checkpoints)
    else:
        checkpoint_set = None
    previous_agent = _copy_agent_for_checkpoint(model) if checkpoint_set else None

    for i in range(num_games):
        game = Game(size=bs)
        model.reset_episode()

        # Randomly assign sides
        if self_play:
            td_player = 0  # plays both sides
        else:
            td_player = random.choice([1, 2])

        while not game.is_over():
            current = game.current_player
            if self_play or current == td_player:
                move = model.choose_move(game)
            else:
                move = opponent.choose_move(game)
            if move is None:
                break
            game.make_move(move[0], move[1])

        # Final update
        if model._prev_cache is not None:
            if game.winner == 0:
                target = 0.5
            elif self_play:
                target = 0.0  # last prev was the loser's perspective
            else:
                # If TD agent won
                target = 1.0 if game.winner == td_player else 0.0
            td_error = target - model._prev_value
            model.update(td_error)

        model.reset_episode()

        if (i + 1) % _CHECKPOINT_EVERY_N_GAMES == 0:
            print(f"  Training game {i + 1}/{num_games}", file=out, flush=True)

        if checkpoint_set is not None and (i + 1) in checkpoint_set:
            _run_checkpoint_evaluations(model, previous_agent, board_size, i + 1, out)
            previous_agent = _copy_agent_for_checkpoint(model)

    model.training = False

# ==============================================================================
# Standalone `training.py` logic
# ==============================================================================

import os
import subprocess
import argparse

# Ensure the logic directory is importable
_logic_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _logic_dir)

# Build Cython modules (only when this file is the entrypoint)
if __name__ == "__main__":
    print("Compiling Cython modules...")
    subprocess.check_call(
        [sys.executable, "setup_cython.py", "build_ext", "--inplace"],
        cwd=_logic_dir,
    )
    print("Cython modules ready.")

from tee import Tee
from td_agent import TDAgent
from td_lambda_agent import TDLambdaAgent
from td_cnn_agent import TDCNNAgent


def main():
    parser = argparse.ArgumentParser(description="Training: train TD agents via self-play or against a given opponent.")
    parser.add_argument("--agent", type=str, default="td",
                        help="Agent to train. Options: td, td_lambda, td_cnn.")
    parser.add_argument("--opponent", type=str, default=None,
                        help="Opponent agent to train against.")
    parser.add_argument("--board-size", type=int, default=7,
                        help="Board size.")
    parser.add_argument("--num-games", type=int, default=2000,
                        help="Number of training games.")
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="Hidden layer size for the agent.")
    parser.add_argument("--td-lambda", type=float, default=0.7,
                        help="Lambda for the TD(λ) agent.")
    parser.add_argument("--checkpoints", type=bool, default=True,
                        help="Enable checkpoint evaluations (enabled by default).")
    parser.add_argument("--save", type=str, default=None,
                        help="Model save path. Relative paths are resolved from the logic directory; .pkl is optional.")
    parser.add_argument("--output", type=str, default=None,
                        help="Also write arena output to this file (still shown on terminal)")
    args = parser.parse_args()

    out = sys.stdout
    out_f = None
    if args.output:
        output_dir = os.path.join(_logic_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, args.output)
        out_f = open(output_path, "w", encoding="utf-8")
        out = Tee(sys.stdout, out_f)
    
    try:
        # Initialize the agent
        model = None
        if args.agent == "td":
            model = TDAgent(board_size=args.board_size, hidden_size=args.hidden_size)
        elif args.agent == "td_lambda":
            model = TDLambdaAgent(board_size=args.board_size, hidden_size=args.hidden_size, lam=args.td_lambda)
        elif args.agent == "td_cnn":
            model = TDCNNAgent(board_size=args.board_size)
        else:
            print(f"Invalid agent: {args.agent}", file=out)
            sys.exit(1)
        
        # Train the agent
        print(f"Training {args.agent} agent ({args.num_games} games)...", file=out)
        train(model, num_games=args.num_games, opponent=args.opponent, board_size=args.board_size, checkpoints=args.checkpoints, out=out)
        print(f"Training complete ({args.num_games} games).", file=out)

        # Save the model
        default_save_path = os.path.join("models", f"{args.agent}_model_s{args.board_size}.pkl")
        requested_save_path = args.save or default_save_path

        save_path = requested_save_path
        if not os.path.isabs(save_path):
            save_path = os.path.join(_logic_dir, save_path)
        if not save_path.lower().endswith(".pkl"):
            save_path = f"{save_path}.pkl"

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        model.save(save_path)
        print(f"Model saved to {save_path}", file=out)
    finally:
        if out_f:
            out_f.close()


if __name__ == "__main__":
    main()
