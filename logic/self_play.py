"""Thread-safe self-play loop with delay for visualization."""

import threading
import time

from game import Game
from mcts import MCTSAgent

BOARD_SIZE = 7

_game = Game(size=BOARD_SIZE)
_agent = MCTSAgent()
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
                game_over = True
            else:
                game_over = False
                move = _agent.choose_move(_game)
                if move is None:
                    break
                _game.make_move(*move)
        if game_over:
            time.sleep(3)
            with _lock:
                _game = Game(size=BOARD_SIZE)
        else:
            time.sleep(0.05)


def start():
    global _running
    if _running:
        return
    _running = True
    t = threading.Thread(target=_play_loop, daemon=True)
    t.start()
