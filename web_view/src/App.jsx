import { useState, useEffect } from "react";
import Board from "./Board";
import "./App.css";

function App() {
  const [state, setState] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const poll = setInterval(() => {
      fetch("http://localhost:5001/state")
        .then((r) => r.json())
        .then((data) => { setState(data); setError(null); })
        .catch(() => setError("Cannot reach server"));
    }, 200);
    fetch("http://localhost:5001/state")
      .then((r) => r.json())
      .then((data) => setState(data))
      .catch(() => setError("Cannot reach server"));
    return () => clearInterval(poll);
  }, []);

  const changeAgent = (player, key) => {
    if (!state?.agents) return;
    const p1 = player === 1 ? key : state.agents["1"].key;
    const p2 = player === 2 ? key : state.agents["2"].key;
    fetch("http://localhost:5001/set_agents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ player1: p1, player2: p2 }),
    });
  };

  const makeMove = (row, col) => {
    fetch("http://localhost:5001/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ row, col }),
    });
  };

  if (error && !state) {
    return <div className="container"><p className="error">{error}</p></div>;
  }
  if (!state) {
    return <div className="container"><p>Loading...</p></div>;
  }

  const agents = state.agents || {};
  const available = state.available_agents || [];
  const p1Label = agents["1"]?.label || "?";
  const p2Label = agents["2"]?.label || "?";
  const isHumanTurn = agents[String(state.current_player)]?.key === "human" && !state.winner;

  const statusText = state.winner
    ? `${state.winner === 1 ? p1Label : p2Label} (${state.winner === 1 ? "Red" : "Blue"}) wins!`
    : isHumanTurn
      ? `Your turn: ${state.current_player === 1 ? "Red" : "Blue"}`
      : `Turn: ${state.current_player === 1 ? p1Label : p2Label} (${state.current_player === 1 ? "Red" : "Blue"})`;

  return (
    <div className="container">
      <h1>Game of Y</h1>

      <div className="matchup-bar">
        <div className="matchup-side red">
          <span className="dot red-dot" />
          <select value={agents["1"]?.key || ""} onChange={(e) => changeAgent(1, e.target.value)}>
            {available.map((a) => (
              <option key={a.key} value={a.key}>{a.label}</option>
            ))}
          </select>
        </div>

        <span className="vs">vs</span>

        <div className="matchup-side blue">
          <select value={agents["2"]?.key || ""} onChange={(e) => changeAgent(2, e.target.value)}>
            {available.map((a) => (
              <option key={a.key} value={a.key}>{a.label}</option>
            ))}
          </select>
          <span className="dot blue-dot" />
        </div>
      </div>

      <p className={state.winner ? "status winner" : "status"}>{statusText}</p>

      <div className="board-wrapper">
        <Board
          size={state.size}
          cells={state.cells}
          moveHistory={state.move_history}
          interactive={isHumanTurn}
          onMove={makeMove}
        />
      </div>

      <p className="move-count">{state.move_history.length} {state.move_history.length === 1 ? "move" : "moves"} played</p>
    </div>
  );
}

export default App;
