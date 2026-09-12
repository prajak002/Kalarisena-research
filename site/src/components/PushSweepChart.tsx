"use client";

import { useState } from "react";
import { PUSH_SWEEP } from "@/lib/content";

const W = 640;
const H = 300;
const PAD = 44;

export function PushSweepChart() {
  const [hover, setHover] = useState<number | null>(null);
  const xs = PUSH_SWEEP.map((d) => d.force);
  const ys = PUSH_SWEEP.map((d) => d.cpMargin);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);

  const x = (v: number) => PAD + ((v - xMin) / (xMax - xMin)) * (W - 2 * PAD);
  const y = (v: number) => H - PAD - ((v - yMin) / (yMax - yMin)) * (H - 2 * PAD);

  const path = PUSH_SWEEP.map((d, idx) => `${idx === 0 ? "M" : "L"}${x(d.force)},${y(d.cpMargin)}`).join(" ");
  const zeroY = y(0);
  const active = hover !== null ? PUSH_SWEEP[hover] : null;

  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="img" aria-label="Capture-point margin after push, by push force">
        <line x1={PAD} y1={zeroY} x2={W - PAD} y2={zeroY} className="chart-zero" />
        <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} className="chart-axis" />
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} className="chart-axis" />
        <path d={path} className="chart-line" fill="none" />
        {PUSH_SWEEP.map((d, idx) => (
          <circle
            key={d.force}
            cx={x(d.force)}
            cy={y(d.cpMargin)}
            r={hover === idx ? 6 : 4}
            className={d.fallRate === 1 ? "chart-dot-fall" : "chart-dot-ok"}
            onMouseEnter={() => setHover(idx)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        <text x={PAD} y={H - PAD + 22} className="chart-tick">0N</text>
        <text x={W - PAD} y={H - PAD + 22} textAnchor="end" className="chart-tick">240N</text>
        <text x={PAD - 8} y={zeroY + 4} textAnchor="end" className="chart-tick">0</text>
      </svg>
      <div className="chart-readout">
        {active ? (
          <>
            <strong>{active.force}N</strong> push - CP margin{" "}
            <strong>{active.cpMargin.toFixed(3)}m</strong> -{" "}
            {active.fallRate === 1 ? "fell (3/3 trials)" : "recovered (0/3 trials fell)"}
          </>
        ) : (
          "Hover a point. Sharp threshold between 100N (recovers) and 120N (falls), 36 trials, scripted controller."
        )}
      </div>
    </div>
  );
}
