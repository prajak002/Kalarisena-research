import type { Metric, StageRow, CodeTraceRow, Status } from "@/lib/content";
import { InlineEq } from "./Math";
import { StatusTag } from "./StatusTag";

export function MetricGrid({ metrics }: { metrics: Metric[] }) {
  return (
    <div className="metric-grid">
      {metrics.map((m) => (
        <div key={m.label} className="metric-card">
          <div className="metric-value">{m.value}</div>
          <div className="metric-label">{m.label}</div>
          {m.detail ? <div className="metric-detail">{m.detail}</div> : null}
        </div>
      ))}
    </div>
  );
}

export function StageTable({ rows }: { rows: StageRow[] }) {
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th>Stage</th>
            <th>File</th>
            <th>What it does</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.stage}>
              <td className="mono">{r.stage}</td>
              <td className="mono">{r.script}</td>
              <td>{r.what}</td>
              <td>
                <StatusTag status={r.status} note={r.note} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function CodeTraceTable({ rows }: { rows: CodeTraceRow[] }) {
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th>Mathematical operation</th>
            <th>Meaning</th>
            <th>Implementation</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.math}>
              <td>
                <InlineEq tex={r.math} />
              </td>
              <td>{r.meaning}</td>
              <td className="mono">
                {r.path}
                {r.symbol ? <div className="mono-sub">:: {r.symbol}</div> : null}
              </td>
              <td>
                <StatusTag status={r.status as Status} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
