"""Thread-safe self-play loop with configurable agents."""

import os
import threading
import time

from game import Game
from mcts import MCTSAgent
from random_agent import RandomAgent
from td_agent import TDAgent
from td_lambda_agent import TDLambdaAgent
from td_cnn_agent import TDCNNAgent
from pv_mcts_agent import (
    PVMCTSAgent,
    load_or_train_pv_mcts,
    load_or_train_self_play_pv_mcts,
)
from sp_pv_cnn_agent import load_or_train_self_play_cnn_pv_mcts

BOARD_SIZE = 7
MOVE_DELAY = 0.3  # seconds between moves for watchability
GAME_OVER_DELAY = 3

# Available agent constructors
AGENT_REGISTRY = {
    "mcts": lambda: MCTSAgent(iterations=5000),
    "pv_mcts": lambda: _load_pv_mcts_agent(),
    "sp_pv_mcts": lambda: _load_self_play_pv_mcts_agent(),
    "sp_pv_cnn": lambda: _load_self_play_pv_cnn_agent(),
    "random": lambda: RandomAgent(),
    "td": lambda: _load_td_agent(),
    "td_lambda": lambda: _load_td_lambda_agent(),
    "td_cnn": lambda: _load_td_cnn_agent(),
}

AGENT_LABELS = {
    "mcts": "MCTS (5k iter)",
    "pv_mcts": "PV-MCTS (400 iter)",
    "sp_pv_mcts": "SP-PV-MCTS (600 iter)",
    "sp_pv_cnn": "SP-PV-CNN (350 iter)",
    "random": "Random",
    "td": "TD(0)",
    "td_lambda": "TD(λ)",
    "td_cnn": "TD-CNN",
}


def _load_td_agent():
    model_path = os.path.join(os.path.dirname(__file__), f"td_model_s{BOARD_SIZE}.pkl")
    if os.path.exists(model_path):
        return TDAgent.load(model_path)
    # No saved model — train a quick one
    print(f"No TD model found at {model_path}, training one...")
    agent = TDAgent(board_size=BOARD_SIZE, hidden_size=128, lr=0.01, epsilon=0.1)
    agent.train(num_games=2000, board_size=BOARD_SIZE)
    agent.save(model_path)
    return agent


def _load_td_lambda_agent():
    model_path = os.path.join(os.path.dirname(__file__), f"td_lambda_model_s{BOARD_SIZE}.pkl")
    if os.path.exists(model_path):
        return TDLambdaAgent.load(model_path)
    print(f"No TD(λ) model found at {model_path}, training one...")
    agent = TDLambdaAgent(board_size=BOARD_SIZE, hidden_size=128, lr=0.01, lam=0.7, epsilon=0.1)
    agent.train(num_games=2000, board_size=BOARD_SIZE)
    agent.save(model_path)
    return agent


def _load_td_cnn_agent():
    model_path = os.path.join(os.path.dirname(__file__), f"td_cnn_model_s{BOARD_SIZE}.pkl")
    if os.path.exists(model_path):
        return TDCNNAgent.load(model_path)
    print(f"No TD-CNN model found at {model_path}, training one...")
    agent = TDCNNAgent(board_size=BOARD_SIZE, lr=0.005, epsilon=0.1)
    agent.train(num_games=2000, board_size=BOARD_SIZE)
    agent.save(model_path)
    return agent


def _load_pv_mcts_agent():
    net, _ = load_or_train_pv_mcts(
        board_size=BOARD_SIZE,
        train_games=200,
        teacher_iters=2000,
        hidden_size=192,
        epochs=80,
    )
    return PVMCTSAgent(net=net, iterations=600)


def _load_self_play_pv_mcts_agent():
    net, _ = load_or_train_self_play_pv_mcts(
        board_size=BOARD_SIZE,
        generations=7,
        games_per_generation=60,
        search_iters=300,
        hidden_size=192,
        epochs=25,
    )
    return PVMCTSAgent(net=net, iterations=600)


def _load_self_play_pv_cnn_agent():
    net, _ = load_or_train_self_play_cnn_pv_mcts(
        board_size=BOARD_SIZE,
        generations=5,
        games_per_generation=20,
        search_iters=100,
        epochs=18,
    )
    return PVMCTSAgent(net=net, iterations=350, rollouts_per_leaf=1, value_weight=0.25)


_game = Game(size=BOARD_SIZE)
_agents = {1: MCTSAgent(iterations=5000), 2: MCTSAgent(iterations=5000)}
_agent_names = {1: "mcts", 2: "mcts"}
_lock = threading.Lock()
_running = False


def get_state():
    with _lock:
        d = _game.to_dict()
        d["agents"] = {
            "1": {"key": _agent_names[1], "label": AGENT_LABELS[_agent_names[1]]},
            "2": {"key": _agent_names[2], "label": AGENT_LABELS[_agent_names[2]]},
        }
        d["available_agents"] = [
            {"key": k, "label": v} for k, v in AGENT_LABELS.items()
        ]
        return d


def set_agents(p1_key, p2_key):
    """Swap in new agents and restart the game."""
    global _game, _agents, _agent_names
    if p1_key not in AGENT_REGISTRY or p2_key not in AGENT_REGISTRY:
        return False
    with _lock:
        _agent_names[1] = p1_key
        _agent_names[2] = p2_key
        _agents[1] = AGENT_REGISTRY[p1_key]()
        _agents[2] = AGENT_REGISTRY[p2_key]()
        _game = Game(size=BOARD_SIZE)
    return True


def _play_loop():
    global _game
    while True:
        with _lock:
            if _game.is_over():
                game_over = True
            else:
                game_over = False
                agent = _agents[_game.current_player]
                move = agent.choose_move(_game)
                if move is None:
                    break
                _game.make_move(*move)
        if game_over:
            time.sleep(GAME_OVER_DELAY)
            with _lock:
                _game = Game(size=BOARD_SIZE)
        else:
            time.sleep(MOVE_DELAY)


def start():
    global _running
    if _running:
        return
    _running = True
    t = threading.Thread(target=_play_loop, daemon=True)
    t.start()
