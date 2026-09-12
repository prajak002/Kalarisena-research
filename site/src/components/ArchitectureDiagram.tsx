"use client";

import katex from "katex";

interface Node {
  id: string;
  label: string;
  eq?: string;
  status: "repo-real" | "paper" | "repo-stub";
  col: number;
  row: number;
}

const NODES: Node[] = [
  { id: "video", label: "Human video", status: "repo-real", col: 0, row: 1 },
  { id: "q0", label: "Raw GEM-X retarget", eq: "Q^0_{1:T}", status: "repo-real", col: 1, row: 1 },
  { id: "proj", label: "Physics-grounded projection", eq: "Q^*_{1:T}", status: "repo-real", col: 2, row: 1 },
  { id: "z", label: "Structured skill state", eq: "z_t", status: "paper", col: 3, row: 0 },
  { id: "scvc", label: "Successor-conditioned viability critic", eq: "V_\\psi", status: "repo-stub", col: 3, row: 2 },
  { id: "residual", label: "Intent-preserving residual", eq: "\\Delta a_t", status: "repo-stub", col: 4, row: 1 },
  { id: "deploy", label: "Deployed action → robot", eq: "a_t^{deploy}", status: "repo-real", col: 5, row: 1 },
];

const EDGES: [string, string][] = [
  ["video", "q0"],
  ["q0", "proj"],
  ["proj", "z"],
  ["proj", "scvc"],
  ["z", "residual"],
  ["scvc", "residual"],
  ["residual", "deploy"],
];

const COL_W = 168;
const ROW_H = 96;
const NODE_W = 140;
const NODE_H = 64;
const PAD = 24;

function center(n: Node) {
  return { x: PAD + n.col * COL_W + NODE_W / 2, y: PAD + n.row * ROW_H + NODE_H / 2 };
}

function renderEq(tex: string) {
  return katex.renderToString(tex, { throwOnError: false, displayMode: false });
}

export function ArchitectureDiagram({ activeId }: { activeId?: string }) {
  const width = PAD * 2 + 6 * COL_W;
  const height = PAD * 2 + 3 * ROW_H;

  return (
    <div className="arch-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} className="arch-svg" role="img" aria-label="KalariSena pipeline architecture">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" className="arch-arrowhead" />
          </marker>
        </defs>
        {EDGES.map(([a, b]) => {
          const na = NODES.find((n) => n.id === a)!;
          const nb = NODES.find((n) => n.id === b)!;
          const ca = center(na);
          const cb = center(nb);
          const x1 = ca.x + NODE_W / 2;
          const x2 = cb.x - NODE_W / 2;
          const y1 = ca.y;
          const y2 = cb.y;
          const mx = (x1 + x2) / 2;
          return (
            <path
              key={`${a}-${b}`}
              d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`}
              className="arch-edge"
              markerEnd="url(#arrow)"
            />
          );
        })}
        {NODES.map((n) => {
          const c = center(n);
          const active = n.id === activeId;
          return (
            <g key={n.id} transform={`translate(${c.x - NODE_W / 2}, ${c.y - NODE_H / 2})`}>
              <rect
                width={NODE_W}
                height={NODE_H}
                rx={5}
                className={`arch-node arch-node-${n.status} ${active ? "arch-node-active" : ""}`}
              />
              <foreignObject x={6} y={6} width={NODE_W - 12} height={NODE_H - 12}>
                <div className="arch-node-body">
                  <div className="arch-node-label">{n.label}</div>
                  {n.eq ? (
                    <div className="arch-node-eq" dangerouslySetInnerHTML={{ __html: renderEq(n.eq) }} />
                  ) : null}
                </div>
              </foreignObject>
            </g>
          );
        })}
      </svg>
      <div className="arch-legend">
        <span className="arch-legend-item"><i className="arch-swatch arch-swatch-repo-real" /> implemented in repo</span>
        <span className="arch-legend-item"><i className="arch-swatch arch-swatch-repo-stub" /> not implemented</span>
        <span className="arch-legend-item"><i className="arch-swatch arch-swatch-paper" /> paper concept</span>
      </div>
    </div>
  );
}
