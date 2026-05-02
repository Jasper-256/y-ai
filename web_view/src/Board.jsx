import React from "react";

const R = 22; // hex radius (in viewBox units)
const SQRT3 = Math.sqrt(3);

function hexCorners(cx, cy) {
  const corners = [];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 180) * (60 * i - 30);
    corners.push(`${cx + R * Math.cos(angle)},${cy + R * Math.sin(angle)}`);
  }
  return corners.join(" ");
}

function cellColor(value) {
  if (value === 1) return "#e74c3c";
  if (value === 2) return "#3498db";
  return "#bdc3c7";
}

export default function Board({ size, cells, moveHistory, interactive, onMove }) {
  const maxRow = size - 1;
  const padX = 30;
  const padY = 25;

  const lastMove =
    moveHistory && moveHistory.length > 0
      ? moveHistory[moveHistory.length - 1]
      : null;

  const hexes = [];
  for (let r = 0; r <= maxRow; r++) {
    for (let c = 0; c <= r; c++) {
      const cx = padX + c * R * SQRT3 + (maxRow - r) * R * SQRT3 * 0.5;
      const cy = padY + r * R * 1.5;
      const key = `${r},${c}`;
      const value = cells[key] || 0;
      const isLast = lastMove && lastMove.row === r && lastMove.col === c;
      const canPlay = interactive && value === 0;

      hexes.push(
        <polygon
          key={key}
          className={canPlay ? "cell playable" : "cell"}
          points={hexCorners(cx, cy)}
          fill={cellColor(value)}
          stroke={isLast ? "#f1c40f" : "#7f8c8d"}
          strokeWidth={isLast ? 3 : 1}
          opacity={value === 0 ? 0.6 : 1}
          onClick={canPlay ? () => onMove(r, c) : undefined}
        />
      );
    }
  }

  const vbWidth = padX * 2 + maxRow * R * SQRT3;
  const vbHeight = padY * 2 + maxRow * R * 1.5;

  return (
    <svg
      viewBox={`0 0 ${vbWidth} ${vbHeight}`}
      preserveAspectRatio="xMidYMid meet"
      style={{ width: "100%", height: "100%", display: "block" }}
    >
      {hexes}
    </svg>
  );
}
