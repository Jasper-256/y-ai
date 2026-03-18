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
        .then((data) => {
          setState(data);
          setError(null);
        })
        .catch(() => setError("Cannot reach server"));
    }, 1000);
    // Initial fetch
    fetch("http://localhost:5001/state")
      .then((r) => r.json())
      .then((data) => setState(data))
      .catch(() => setError("Cannot reach server"));
    return () => clearInterval(poll);
  }, []);

  if (error && !state) {
    return <div className="container"><p className="error">{error}</p></div>;
  }
  if (!state) {
    return <div className="container"><p>Loading...</p></div>;
  }

  const statusText = state.winner
    ? `Player ${state.winner} (${state.winner === 1 ? "Red" : "Blue"}) wins!`
    : `Turn: Player ${state.current_player} (${state.current_player === 1 ? "Red" : "Blue"}) / Move ${state.move_history.length + 1}`;

  return (
    <div className="container">
      <h1>Game of Y Visualization</h1>
      <p className={state.winner ? "status winner" : "status"}>{statusText}</p>
      <div className="board-wrapper">
        <Board
          size={state.size}
          cells={state.cells}
          moveHistory={state.move_history}
        />
      </div>
      <p className="move-count">{state.move_history.length} moves played</p>
    </div>
  );
}

export default App;
