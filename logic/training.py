"""Unified training loop for all TD agents.

Optional training checkpoints:
at each specified game index, run three mini-tournaments: 
- current vs random
- current vs heuristic
- current vs the snapshot of the prior checkpoint
"""

import copy
import random
import sys

# Progress log interval and default spacing when checkpoints=True.
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
            run_tournament(agents, games_per_matchup=games_per_matchup, board_size=bs, out=out)
    finally:
        model.training = was_training


def train(model, num_games=1000, opponent=None, board_size=None, checkpoints=None):
    """Train via self-play or against a given opponent.

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
            print(f"  Training game {i + 1}/{num_games}")

        if checkpoint_set is not None and (i + 1) in checkpoint_set:
            _run_checkpoint_evaluations(model, previous_agent, board_size, i + 1)
            previous_agent = _copy_agent_for_checkpoint(model)

    model.training = False
