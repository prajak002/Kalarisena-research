"use client";

import { useState } from "react";
import { SOMA_STAGES } from "@/lib/content";

export function SomaPipeline() {
  const [idx, setIdx] = useState(0);
  const stage = SOMA_STAGES[idx];

  return (
    <div className="video-toggle">
      <video
        key={stage.file}
        className="video-frame"
        src={`/media/videos/soma/${stage.file}`}
        controls
        loop
        muted
        playsInline
      />
      <div className="step-rail" style={{ marginTop: 14, marginBottom: 10 }}>
        {SOMA_STAGES.map((s, i) => (
          <button
            key={s.key}
            className={`step-dot ${i === idx ? "active" : ""} ${i < idx ? "done" : ""}`}
            onClick={() => setIdx(i)}
            aria-label={s.label}
          />
        ))}
      </div>
      <div className="video-toggle-tabs">
        {SOMA_STAGES.map((s, i) => (
          <button
            key={s.key}
            className={`tab-btn ${i === idx ? "active" : ""}`}
            onClick={() => setIdx(i)}
          >
            {s.label}
          </button>
        ))}
      </div>
      <p className="figure-caption">{stage.caption}</p>
    </div>
  );
}
