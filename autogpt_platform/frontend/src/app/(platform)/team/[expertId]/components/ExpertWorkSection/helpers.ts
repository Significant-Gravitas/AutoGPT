export interface RunStatusInfo {
  label: string;
  className: string;
}

const STATUS_INFO: Record<string, RunStatusInfo> = {
  COMPLETED: {
    label: "Completed",
    className: "bg-emerald-50 text-emerald-600",
  },
  FAILED: { label: "Failed", className: "bg-red-50 text-red-600" },
  RUNNING: { label: "Running", className: "bg-blue-50 text-blue-600" },
  QUEUED: { label: "Queued", className: "bg-zinc-100 text-zinc-600" },
  REVIEW: {
    label: "Waiting for review",
    className: "bg-amber-50 text-amber-700",
  },
  TERMINATED: { label: "Stopped", className: "bg-zinc-100 text-zinc-600" },
  INCOMPLETE: { label: "Incomplete", className: "bg-zinc-100 text-zinc-600" },
};

/**
 * Honest chip per execution status — a queued or running run must never
 * read as "Completed". Unrecognized statuses fall back to showing the raw
 * value in neutral styling rather than guessing an outcome.
 */
export function getRunStatusInfo(status: string): RunStatusInfo {
  return (
    STATUS_INFO[status.toUpperCase()] ?? {
      label: status,
      className: "bg-zinc-100 text-zinc-600",
    }
  );
}
