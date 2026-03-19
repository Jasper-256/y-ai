"""Flask API serving game state for the React frontend."""

import subprocess
import sys
import os

# ── Compile Cython MCTS before importing anything that needs it ──
_logic_dir = os.path.dirname(os.path.abspath(__file__))
print("Compiling Cython MCTS module…")
subprocess.check_call(
    [sys.executable, "setup_cython.py", "build_ext", "--inplace"],
    cwd=_logic_dir,
)
print("Cython MCTS module ready.")

from flask import Flask, jsonify
from flask_cors import CORS

import self_play

app = Flask(__name__)
CORS(app)


@app.route("/state")
def state():
    return jsonify(self_play.get_state())


if __name__ == "__main__":
    self_play.start()
    app.run(port=5001)
