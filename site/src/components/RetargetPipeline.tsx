"use client";

import { useState } from "react";
import { RETARGET_MOTIONS, RETARGET_STAGES } from "@/lib/content";

export function RetargetPipeline() {
  const [motionIdx, setMotionIdx] = useState(0);
  const [stageIdx, setStageIdx] = useState(0);
  const motion = RETARGET_MOTIONS[motionIdx];
  const stage = RETARGET_STAGES[stageIdx];
  const src = `/media/videos/retarget/${stage.key}_${motion.id}.mp4`;

  return (
    <div className="video-toggle">
      <div className="video-toggle-tabs">
        {RETARGET_MOTIONS.map((m, idx) => (
          <button
            key={m.id}
            className={`tab-btn ${idx === motionIdx ? "active" : ""}`}
            onClick={() => setMotionIdx(idx)}
          >
            {m.label}
          </button>
        ))}
      </div>

      <video key={src} className="video-frame" src={src} controls loop muted playsInline />

      <div className="step-rail" style={{ marginTop: 14, marginBottom: 10 }}>
        {RETARGET_STAGES.map((s, idx) => (
          <button
            key={s.key}
            className={`step-dot ${idx === stageIdx ? "active" : ""} ${idx < stageIdx ? "done" : ""}`}
            onClick={() => setStageIdx(idx)}
            aria-label={s.label}
          />
        ))}
      </div>
      <div className="video-toggle-tabs">
        {RETARGET_STAGES.map((s, idx) => (
          <button
            key={s.key}
            className={`tab-btn ${idx === stageIdx ? "active" : ""}`}
            onClick={() => setStageIdx(idx)}
          >
            {s.label}
          </button>
        ))}
      </div>
      <p className="figure-caption">{stage.caption}</p>
      <p className="figure-caption">
        Motion family (from configs/motion_families.yaml): <code className="code-pill">{motion.family}</code>
      </p>
    </div>
  );
}
