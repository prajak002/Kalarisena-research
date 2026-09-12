"use client";

import { useEffect, useState } from "react";
import type { WalkStep } from "@/lib/content";
import { Eq } from "./Math";
import { StatusTag } from "./StatusTag";

export function StepWalkthrough({ steps }: { steps: WalkStep[] }) {
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const step = steps[i];

  useEffect(() => {
    if (!playing) return;
    const t = setTimeout(() => {
      setI((p) => {
        if (p >= steps.length - 1) {
          setPlaying(false);
          return p;
        }
        return p + 1;
      });
    }, 3600);
    return () => clearTimeout(t);
  }, [playing, i, steps.length]);

  return (
    <div className="walk">
      <div className="walk-controls">
        <button
          className="ctrl-btn"
          onClick={() => setPlaying((p) => !p)}
          aria-label={playing ? "Pause" : "Step through automatically"}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button
          className="ctrl-btn"
          onClick={() => setI((p) => Math.min(p + 1, steps.length - 1))}
        >
          Step
        </button>
        <button
          className="ctrl-btn"
          onClick={() => {
            setI(0);
            setPlaying(false);
          }}
        >
          Restart
        </button>
        <span className="step-counter">
          Step {i + 1} / {steps.length}
        </span>
      </div>

      <div className="step-rail">
        {steps.map((s, idx) => (
          <button
            key={s.id}
            className={`step-dot ${idx === i ? "active" : ""} ${idx < i ? "done" : ""}`}
            onClick={() => {
              setPlaying(false);
              setI(idx);
            }}
            aria-label={s.title}
          />
        ))}
      </div>

      <h4 className="walk-title">{step.title}</h4>

      <div className="walk-grid">
        <div className="walk-panel">
          <div className="panel-label">Visual</div>
          <p className="panel-body">{step.visual}</p>
        </div>
        <div className="walk-panel">
          <div className="panel-label">Computation</div>
          <p className="panel-body">{step.computation}</p>
        </div>
        <div className="walk-panel walk-panel-math">
          <div className="panel-label">Mathematics</div>
          <Eq tex={step.math} />
        </div>
      </div>

      <div className="walk-footer">
        <p className="walk-explanation">{step.explanation}</p>
        <div className="walk-meta">
          <StatusTag status={step.status} note={step.statusNote} />
          {step.code ? (
            <code className="code-pill">
              {step.code.path}
              {step.code.symbol ? ` :: ${step.code.symbol}` : ""}
            </code>
          ) : null}
        </div>
      </div>
    </div>
  );
}
