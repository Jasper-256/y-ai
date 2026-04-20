# Y AI

Our project repo for CMPE-260

Group members: Jasper Morgal, Advait Shinde, Nickzad Bayati, Toney Zhen

## About

Game of Y is a connection board game played on a triangular hex grid. Two players (Red and Blue) take turns placing stones. The goal is to form a connected group of your stones that touches all three sides of the triangle. Y cannot end in a draw.

This project implements:
- A Python game engine with a triangular hex board, adjacency logic, and BFS-based win detection
- An MCTS (Monte Carlo Tree Search) AI agent using UCB1 selection and random rollouts
- A TD(0) (Temporal Difference) AI agent with a neural network value function
- A TD(λ) agent extending TD(0) with eligibility traces for multi-step credit assignment
- A TD-CNN agent that replaces the TD(0) MLP with a convolutional network over a diagonal 2D embedding of the triangular board
- A heuristic baseline agent using 1-ply lookahead and static evaluation
- A random baseline agent
- An arena for running tournaments between agents and recording win rates
- A Flask API that serves the live game state
- A React frontend that visualizes games in real-time with a matchup picker to watch different agents play each other

## Agents

Each agent lives in `logic/` and exposes a `choose_move(game)` method.

- **Random** (`random_agent.py`) — picks a uniformly random legal move. A trivial baseline.
- **Heuristic** (`heuristic_agent.py`) — non-learning 1-ply lookahead that scores each successor by how many of the triangle's three sides its best connected group touches (primary), stone count, and adjacent empty cells (tiebreakers).
- **MCTS** (`mcts.pyx`) — Monte Carlo Tree Search with UCB1 selection (c=1.41) and uniform-random rollouts, budgeted by iterations per move (default 1000, `--mcts-iters`). Implemented in Cython for speed.
- **TD(0)** (`td_agent.py`) — small MLP value function `V(s) ∈ [0, 1]` trained via self-play TD(0) updates; picks moves with 1-ply lookahead, choosing the child that minimizes the opponent's predicted win probability.
- **TD(λ)** (`td_lambda_agent.py`) — same network and selection rule as TD(0), but uses eligibility traces so a single TD error propagates credit back across many prior states (`--td-lambda`, default 0.7).
- **TD-CNN** (`td_cnn_agent.py`) — TD(0) with a CNN value function. The triangular board is embedded diagonally into an `N × N` grid and fed in as 4 channels (exists mask, mine, opponent, empty) so the network can see the board shape. Three 3×3 conv layers feed a small dense head.

## Setup

### Backend

```bash
cd logic
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend

```bash
cd web_view
npm install
```

## Running

Start the backend and frontend in separate terminals:

**Terminal 1: API server:**
```bash
cd logic
source venv/bin/activate
python server.py
```

**Terminal 2: React dev server:**
```bash
cd web_view
npm run dev
```

Open http://localhost:5173 to watch agents play. Use the dropdowns to pick which agents play as Red and Blue (MCTS, TD(0), TD(λ), TD-CNN, or Random).

### Arena

The arena runs a round-robin tournament between agents and prints win rates, head-to-head results, and first-player advantage stats.

```bash
cd logic
source venv/bin/activate

# Run all three agents against each other (100 games per matchup)
python arena.py

# Just Random vs TD
python arena.py --agents random td

# Include TD(λ) in the tournament
python arena.py --agents random td td_lambda

# Include the TD-CNN agent
python arena.py --agents random td td_cnn

# Customize settings
python arena.py --games 200 --size 5 --mcts-iters 1000

# Train a fresh TD model (instead of loading the saved one)
python arena.py --agents random td --td-retrain --td-train 10000

# Set a custom lambda value for TD(λ) (default: 0.7)
python arena.py --agents td td_lambda --td-lambda 0.9

# Change the hidden layer size for TD agents (default: 128)
python arena.py --agents td td_lambda --td-hidden 256 --td-retrain

# Load a specific TD model
python arena.py --td-model path/to/model.pkl
```
