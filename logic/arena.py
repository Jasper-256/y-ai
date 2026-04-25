#!/usr/bin/env python3
"""Arena: pit agents against each other and record win rates.

Usage:
    python arena.py                          # default matchups
    python arena.py --games 200 --size 5     # custom settings
    python arena.py --output results.txt     # tee report to terminal and file

The arena plays N games for each ordered pair (agent_as_P1, agent_as_P2),
so each pair plays 2*N games total (N with each side).
"""

import sys
import os
import subprocess
import time
import argparse
from collections import defaultdict

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

from game import Game
from mcts import MCTSAgent
from random_agent import RandomAgent
from heuristic_agent import HeuristicAgent
from td_agent import TDAgent
from td_lambda_agent import TDLambdaAgent
from td_cnn_agent import TDCNNAgent
from tee import Tee


def play_game(agent1, agent2, board_size=7):
    """Play a single game. agent1 is player 1 (Red), agent2 is player 2 (Blue).

    Returns the winner (1 or 2), or 0 for no result.
    """
    game = Game(size=board_size)
    agents = {1: agent1, 2: agent2}

    while not game.is_over():
        agent = agents[game.current_player]
        move = agent.choose_move(game)
        if move is None:
            break
        game.make_move(move[0], move[1])

    return game.winner


def run_tournament(agents, games_per_matchup=100, board_size=7, out=sys.stdout):
    """Run a round-robin tournament between named agents.

    Args:
        agents: dict of {name: agent_instance}
        games_per_matchup: Games per ordered pair (each agent plays this many
                           games as P1 against each opponent).
        board_size: Board size for all games.

    Returns:
        results: dict of (name1, name2) -> {1: wins_for_name1, 2: wins_for_name2}
    """
    names = list(agents.keys())
    results = {}

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i == j:
                continue

            key = (name1, name2)
            wins = {1: 0, 2: 0}
            agent1 = agents[name1]
            agent2 = agents[name2]

            print(f"\n{name1} (P1) vs {name2} (P2): ", end="", flush=True, file=out)
            t0 = time.time()

            for g in range(games_per_matchup):
                winner = play_game(agent1, agent2, board_size)
                if winner in (1, 2):
                    wins[winner] += 1
                if (g + 1) % 10 == 0:
                    print(".", end="", flush=True, file=out)

            elapsed = time.time() - t0
            print(f" done ({elapsed:.1f}s)", file=out)
            print(f"  {name1} wins: {wins[1]}, {name2} wins: {wins[2]}", file=out)
            results[key] = wins

    return results


def print_summary(agents, results, games_per_matchup, out=sys.stdout):
    """Print a summary table of overall win rates."""
    names = list(agents.keys())

    # Aggregate: total wins and total games for each agent
    total_wins = defaultdict(int)
    total_games = defaultdict(int)

    # Head-to-head matrix
    h2h = {}
    for name in names:
        h2h[name] = {}

    for (n1, n2), wins in results.items():
        total_wins[n1] += wins[1]
        total_wins[n2] += wins[2]
        total_games[n1] += wins[1] + wins[2]
        total_games[n2] += wins[1] + wins[2]
        # n1 was P1, n2 was P2
        h2h[n1][n2] = h2h.get(n1, {}).get(n2, 0) + wins[1]
        h2h[n2][n1] = h2h.get(n2, {}).get(n1, 0) + wins[2]

    # Print overall standings
    print("\n" + "=" * 60, file=out)
    print("OVERALL STANDINGS", file=out)
    print("=" * 60, file=out)
    standings = []
    for name in names:
        tg = total_games[name]
        tw = total_wins[name]
        rate = tw / tg if tg > 0 else 0.0
        standings.append((name, tw, tg, rate))

    standings.sort(key=lambda x: -x[3])
    print(f"{'Agent':<20} {'Wins':>6} {'Games':>7} {'Win%':>7}", file=out)
    print("-" * 42, file=out)
    for name, tw, tg, rate in standings:
        print(f"{name:<20} {tw:>6} {tg:>7} {rate:>6.1%}", file=out)

    # Print head-to-head matrix
    print("\n" + "=" * 60, file=out)
    print("HEAD-TO-HEAD (row vs column, total wins across both sides)", file=out)
    print("=" * 60, file=out)

    col_width = max(len(n) for n in names) + 2
    header = " " * col_width
    for n in names:
        header += f"{n:>{col_width}}"
    print(header, file=out)

    for n1 in names:
        row = f"{n1:<{col_width}}"
        for n2 in names:
            if n1 == n2:
                row += f"{'---':>{col_width}}"
            else:
                w = h2h.get(n1, {}).get(n2, 0)
                total = w + h2h.get(n2, {}).get(n1, 0)
                if total > 0:
                    row += f"{f'{w}/{total}':>{col_width}}"
                else:
                    row += f"{'N/A':>{col_width}}"
        print(row, file=out)

    # Print P1/P2 advantage
    print("\n" + "=" * 60, file=out)
    print("FIRST-PLAYER ADVANTAGE (per matchup)", file=out)
    print("=" * 60, file=out)
    print(f"{'Matchup':<30} {'P1 wins':>8} {'P2 wins':>8} {'P1 rate':>8}", file=out)
    print("-" * 56, file=out)
    for (n1, n2), wins in sorted(results.items()):
        total = wins[1] + wins[2]
        p1_rate = wins[1] / total if total > 0 else 0.0
        print(f"{n1} vs {n2:<18} {wins[1]:>8} {wins[2]:>8} {p1_rate:>7.1%}", file=out)


def main():
    parser = argparse.ArgumentParser(description="Arena: pit Game of Y agents against each other")
    parser.add_argument("--games", type=int, default=100,
                        help="Games per matchup direction (default: 100)")
    parser.add_argument("--size", type=int, default=7,
                        help="Board size (default: 7)")
    parser.add_argument("--mcts-iters", type=int, default=1000,
                        help="MCTS iterations per move (default: 1000)")
    parser.add_argument("--td-train", type=int, default=2000,
                        help="Number of self-play games to train the TD agent (default: 2000)")
    parser.add_argument("--td-hidden", type=int, default=128,
                        help="Hidden layer size for TD agents (default: 128)")
    parser.add_argument("--td-retrain", action="store_true",
                        help="Force training a new TD model even if one exists")
    parser.add_argument("--td-model", type=str, default=None,
                        help="Path to a saved TD model (skip training)")
    parser.add_argument("--td-lambda", type=float, default=0.7,
                        help="Lambda value for the TD(λ) agent (default: 0.7)")
    parser.add_argument("--td-lambda-model", type=str, default=None,
                        help="Path to a saved TD(λ) model (skip training)")
    parser.add_argument("--td-cnn-model", type=str, default=None,
                        help="Path to a saved TD-CNN model (skip training)")
    parser.add_argument("--agents", type=str, nargs="+",
                        default=["random", "heuristic", "td", "td_lambda", "td_cnn", "mcts"],
                        help="Which agents to include: random, heuristic, td, td_lambda, td_cnn, mcts (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Also write arena output to this file (still shown on terminal)")
    args = parser.parse_args()

    out = sys.stdout
    out_f = None
    if args.output:
        out_f = open(args.output, "w", encoding="utf-8")
        out = Tee(sys.stdout, out_f)

    try:
        agents = {}

        if "random" in args.agents:
            agents["Random"] = RandomAgent()
            print("Loaded Random agent.", file=out)

        if "heuristic" in args.agents:
            agents["Heuristic"] = HeuristicAgent()
            print("Loaded Heuristic agent.")

        if "td" in args.agents:
            model_path = args.td_model or os.path.join(_logic_dir, f"td_model_s{args.size}.pkl")
            if not args.td_retrain and os.path.exists(model_path):
                td = TDAgent.load(model_path)
                print(f"Loaded TD agent from {model_path}", file=out)
            else:
                print(f"Training TD agent ({args.td_train} self-play games on size {args.size} board)...", file=out)
                td = TDAgent(board_size=args.size, hidden_size=args.td_hidden, lr=0.01, epsilon=0.1)
                td.train(num_games=args.td_train, board_size=args.size)
                save_path = args.td_model or os.path.join(_logic_dir, f"td_model_s{args.size}.pkl")
                td.save(save_path)
            agents["TD(0)"] = td

        if "td_lambda" in args.agents:
            model_path = args.td_lambda_model or os.path.join(_logic_dir, f"td_lambda_model_s{args.size}.pkl")
            if not args.td_retrain and os.path.exists(model_path):
                td_lam = TDLambdaAgent.load(model_path)
                print(f"Loaded TD(λ) agent from {model_path}", file=out)
            else:
                print(f"Training TD(λ) agent ({args.td_train} self-play games on size {args.size} board, λ={args.td_lambda})...", file=out)
                td_lam = TDLambdaAgent(board_size=args.size, hidden_size=args.td_hidden, lr=0.01, lam=args.td_lambda, epsilon=0.1)
                td_lam.train(num_games=args.td_train, board_size=args.size)
                save_path = args.td_lambda_model or os.path.join(_logic_dir, f"td_lambda_model_s{args.size}.pkl")
                td_lam.save(save_path)
            agents[f"TD(λ={args.td_lambda})"] = td_lam

        if "td_cnn" in args.agents:
            model_path = args.td_cnn_model or os.path.join(_logic_dir, f"td_cnn_model_s{args.size}.pkl")
            if not args.td_retrain and os.path.exists(model_path):
                td_cnn = TDCNNAgent.load(model_path)
                print(f"Loaded TD-CNN agent from {model_path}", file=out)
            else:
                print(f"Training TD-CNN agent ({args.td_train} self-play games on size {args.size} board)...", file=out)
                td_cnn = TDCNNAgent(board_size=args.size, lr=0.005, epsilon=0.1)
                td_cnn.train(num_games=args.td_train, board_size=args.size)
                save_path = args.td_cnn_model or os.path.join(_logic_dir, f"td_cnn_model_s{args.size}.pkl")
                td_cnn.save(save_path)
            agents["TD-CNN"] = td_cnn

        if "mcts" in args.agents:
            agents[f"MCTS({args.mcts_iters})"] = MCTSAgent(iterations=args.mcts_iters)
            print(f"Loaded MCTS agent ({args.mcts_iters} iterations).", file=out)

        if len(agents) < 2:
            print("Need at least 2 agents. Use --agents to specify.", file=out)
            sys.exit(1)

        print(f"\nStarting tournament: {list(agents.keys())}", file=out)
        print(f"Board size: {args.size}, Games per matchup: {args.games}", file=out)
        print("=" * 60, file=out)

        results = run_tournament(agents, games_per_matchup=args.games, board_size=args.size, out=out)
        print_summary(agents, results, args.games, out=out)
    finally:
        if out_f:
            out_f.close()


if __name__ == "__main__":
    main()
