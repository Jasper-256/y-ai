# Y AI

Our project repo for CMPE-260

Group members: Jasper Morgal, Advait Shinde, Nickzad Bayati, Toney Zhen

## About

Game of Y is a connection board game played on a triangular hex grid. Two players (Red and Blue) take turns placing stones. The goal is to form a connected group of your stones that touches all three sides of the triangle. Y cannot end in a draw.

This project implements:
- A Python game engine with a triangular hex board, adjacency logic, and BFS-based win detection
- An MCTS (Monte Carlo Tree Search) AI agent that plays against itself using UCB1 selection and random rollouts
- A Flask API that serves the live game state
- A React frontend that visualizes the board in real-time as the AI plays

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

Open http://localhost:5173 to watch the agent play against itself.

## Project Structure

```
logic/
  board.py          # Board representation, adjacency, win detection (BFS)
  game.py           # Game state: turn tracking, move application, serialization
  mcts.py           # MCTS agent: UCB1 selection, expansion, random rollout
  self_play.py      # Thread-safe self-play loop with delay for visualization
  server.py         # Flask API: GET /state endpoint
  requirements.txt  # Python dependencies

web_view/           # Vite + React app
  src/
    App.jsx         # Polls /state every 1s, displays turn/winner
    Board.jsx       # SVG triangular hex board renderer
```
