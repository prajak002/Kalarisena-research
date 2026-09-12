import type { RoadmapStage } from "@/lib/content";

export function Roadmap({ stages }: { stages: RoadmapStage[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {stages.map((s) => (
        <div key={s.stage} className="walk-panel" style={{ padding: 20 }}>
          <div className="panel-label">{s.stage}</div>
          <h4 className="walk-title" style={{ marginBottom: 10 }}>{s.title}</h4>
          <p className="panel-body" style={{ marginBottom: 10 }}>
            <strong>Problem it solves: </strong>{s.problem}
          </p>
          <p className="panel-body" style={{ marginBottom: 10 }}>
            <strong>Approach: </strong>{s.approach}
          </p>
          <code className="code-pill">{s.files}</code>
        </div>
      ))}
    </div>
  );
}
