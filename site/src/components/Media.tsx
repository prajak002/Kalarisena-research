"use client";

import { useState } from "react";
import Image from "next/image";

export function Figure({
  src,
  caption,
  width = 1200,
  height = 800,
}: {
  src: string;
  caption: string;
  width?: number;
  height?: number;
}) {
  return (
    <figure className="figure">
      <Image src={src} alt={caption} width={width} height={height} className="figure-img" />
      <figcaption className="figure-caption">{caption}</figcaption>
    </figure>
  );
}

export function VideoToggle({
  options,
}: {
  options: { label: string; src: string; caption: string }[];
}) {
  const [active, setActive] = useState(0);
  const opt = options[active];
  return (
    <div className="video-toggle">
      <div className="video-toggle-tabs">
        {options.map((o, idx) => (
          <button
            key={o.label}
            className={`tab-btn ${idx === active ? "active" : ""}`}
            onClick={() => setActive(idx)}
          >
            {o.label}
          </button>
        ))}
      </div>
      <video
        key={opt.src}
        className="video-frame"
        src={opt.src}
        controls
        loop
        muted
        playsInline
      />
      <p className="figure-caption">{opt.caption}</p>
    </div>
  );
}
