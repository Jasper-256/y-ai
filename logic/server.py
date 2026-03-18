"""Flask API serving game state for the React frontend."""

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
