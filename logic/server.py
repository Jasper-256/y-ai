"""Flask API serving game state for the React frontend."""

import subprocess
import sys
import os

# ── Compile Cython modules before importing anything that needs them ──
_logic_dir = os.path.dirname(os.path.abspath(__file__))
print("Compiling Cython modules…")
subprocess.check_call(
    [sys.executable, "setup_cython.py", "build_ext", "--inplace"],
    cwd=_logic_dir,
)
print("Cython modules ready.")

from flask import Flask, jsonify, request
from flask_cors import CORS

import self_play

app = Flask(__name__)
CORS(app)


@app.route("/state")
def state():
    return jsonify(self_play.get_state())


@app.route("/set_agents", methods=["POST"])
def set_agents():
    data = request.get_json(force=True)
    p1 = data.get("player1", "mcts")
    p2 = data.get("player2", "mcts")
    ok = self_play.set_agents(p1, p2)
    if ok:
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "message": "unknown agent key"}), 400


@app.route("/move", methods=["POST"])
def move():
    data = request.get_json(force=True)
    try:
        row = int(data["row"])
        col = int(data["col"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"status": "error", "message": "row and col are required"}), 400

    ok, message = self_play.make_human_move(row, col)
    if ok:
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "message": message}), 400


if __name__ == "__main__":
    self_play.start()
    app.run(port=5001)
