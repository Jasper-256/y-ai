# Y AI

Our project repo for CMPE-260

Group members: Jasper Morgal, Advait Shinde, Nickzad Bayati, Toney Zhen

## About

Game of Y is a connection board game played on a triangular hex grid. Two players (Red and Blue) take turns placing stones. The goal is to form a connected group of your stones that touches all three sides of the triangle. Y cannot end in a draw.

This project implements:
- A Python game engine with a triangular hex board, adjacency logic, and BFS-based win detection
- An MCTS (Monte Carlo Tree Search) AI agent using UCB1 selection and random rollouts
- A TD(0) (Temporal Difference) AI agent with a neural network value function
- A random baseline agent
- An arena for running tournaments between agents and recording win rates
- A Flask API that serves the live game state
- A React frontend that visualizes games in real-time with a matchup picker to watch different agents play each other

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

Open http://localhost:5173 to watch agents play. Use the dropdowns to pick which agents play as Red and Blue (MCTS, TD(0), or Random).

### Arena

The arena runs a round-robin tournament between agents and prints win rates, head-to-head results, and first-player advantage stats.

```bash
cd logic
source venv/bin/activate

# Run all three agents against each other (100 games per matchup)
python arena.py

# Just Random vs TD
python arena.py --agents random td

# Customize settings
python arena.py --games 200 --size 5 --mcts-iters 1000

# Train a fresh TD model (instead of loading the saved one)
python arena.py --agents random td --td-retrain --td-train 10000

# Load a specific TD model
python arena.py --td-model path/to/model.pkl
```

## Project Structure

```
logic/
  board.pyx         # Board representation, adjacency, win detection (BFS) — Cython
  game.pyx          # Game state: turn tracking, move application, serialization — Cython
  mcts.pyx          # MCTS agent: UCB1 selection, expansion, random rollout — Cython
  td_agent.py       # TD(0) agent: MLP value function, self-play training
  random_agent.py   # Random baseline agent
  arena.py          # Tournament runner: round-robin matchups with win rate stats
  self_play.py      # Thread-safe play loop with configurable agents
  server.py         # Flask API: GET /state, POST /set_agents
  requirements.txt  # Python dependencies

web_view/           # Vite + React app
  src/
    App.jsx         # Polls /state, agent matchup picker
    Board.jsx       # SVG triangular hex board renderer
```
