"""Thread-safe self-play loop with delay for visualization."""

import threading
import time

from game import Game
from mcts import MCTSAgent

_game = Game(size=9)
_agent = MCTSAgent(iterations=800)
_lock = threading.Lock()
_running = False


def get_state():
    with _lock:
        return _game.to_dict()


def _play_loop():
    global _game
    while True:
        with _lock:
            if _game.is_over():
                break
            move = _agent.choose_move(_game)
            if move is None:
                break
            _game.make_move(*move)
        time.sleep(1.5)


def start():
    global _running
    if _running:
        return
    _running = True
    t = threading.Thread(target=_play_loop, daemon=True)
    t.start()
