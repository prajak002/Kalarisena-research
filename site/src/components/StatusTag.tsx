import type { Status } from "@/lib/content";

const LABEL: Record<Status, string> = {
  paper: "Paper's proposed method",
  "repo-real": "Implemented in this repo",
  "repo-stub": "Not yet implemented",
  negative: "Documented negative result",
};

export function StatusTag({ status, note }: { status: Status; note?: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span className={`status-tag status-${status}`}>{LABEL[status]}</span>
      {note ? <span className="text-xs text-neutral-500 leading-snug max-w-sm">{note}</span> : null}
    </div>
  );
}
