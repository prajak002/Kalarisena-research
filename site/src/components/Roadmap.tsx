import Image from "next/image";
import type { RoadmapStage } from "@/lib/content";
import { StatusTag } from "./StatusTag";

export function Roadmap({ stages }: { stages: RoadmapStage[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {stages.map((s) => (
        <div key={s.stage} className="walk-panel" style={{ padding: 20 }}>
          <div className="panel-label">{s.stage}</div>
          <h4 className="walk-title" style={{ marginBottom: 10 }}>{s.title}</h4>
          {s.status ? <div style={{ marginBottom: 10 }}><StatusTag status={s.status} note={s.statusNote} /></div> : null}
          <p className="panel-body" style={{ marginBottom: 10 }}>
            <strong>Problem it solves: </strong>{s.problem}
          </p>
          <p className="panel-body" style={{ marginBottom: 10 }}>
            <strong>Approach: </strong>{s.approach}
          </p>
          {s.figure ? (
            <div style={{ margin: "12px 0" }}>
              <Image src={s.figure} alt={`${s.title} result`} width={820} height={520} className="figure-img" />
            </div>
          ) : null}
          <code className="code-pill">{s.files}</code>
        </div>
      ))}
    </div>
  );
}
