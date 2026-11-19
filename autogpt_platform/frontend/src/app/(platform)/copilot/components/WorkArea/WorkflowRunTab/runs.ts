import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { toMs } from "./trace";

/** Poll this often while a run is still going. */
export const LIVE_POLL_MS = 1500;

const LIVE_STATUSES = new Set(["QUEUED", "RUNNING", "INCOMPLETE"]);

export function isLiveStatus(status: string | undefined) {
  return LIVE_STATUSES.has((status ?? "").toUpperCase());
}

export type RunTone = "live" | "ok" | "failed" | "idle" | "waiting";

const STATUS_VIEW: Record<string, { label: string; tone: RunTone }> = {
  QUEUED: { label: "Queued", tone: "live" },
  INCOMPLETE: { label: "Starting", tone: "live" },
  RUNNING: { label: "Running", tone: "live" },
  COMPLETED: { label: "Done", tone: "ok" },
  FAILED: { label: "Failed", tone: "failed" },
  TERMINATED: { label: "Stopped", tone: "idle" },
  REVIEW: { label: "Waiting for review", tone: "waiting" },
};

export function statusView(status: string | undefined) {
  return (
    STATUS_VIEW[(status ?? "").toUpperCase()] ?? {
      label: status ?? "Unknown",
      tone: "idle" as const,
    }
  );
}

const TRIGGER_LABELS: Record<string, string> = {
  manual: "Run by hand",
  api: "Started through the API",
  schedule: "Scheduled run",
  webhook: "Started by a trigger",
  copilot: "Started by Otto",
  subgraph: "Started by another workflow",
};

export function triggerLabel(run: GraphExecutionMeta) {
  if (run.trigger_source && TRIGGER_LABELS[run.trigger_source]) {
    return TRIGGER_LABELS[run.trigger_source];
  }
  if (run.schedule_id) return TRIGGER_LABELS.schedule;
  if (run.webhook_id) return TRIGGER_LABELS.webhook;
  return "Run";
}

/** How long the run took, or has taken so far. */
export function runElapsedMs(
  run: Pick<GraphExecutionMeta, "started_at" | "ended_at" | "stats">,
  now: number,
) {
  const start = toMs(run.started_at);
  if (start === null) return null;
  const end = toMs(run.ended_at);
  if (end !== null) return Math.max(0, end - start);
  if (run.stats?.duration) return run.stats.duration * 1000;
  return Math.max(0, now - start);
}

/** The run's cost, when it cost anything; stats count cents. */
export function formatCost(cents: number | undefined) {
  if (!cents) return null;
  return `$${(cents / 100).toFixed(cents < 100 ? 3 : 2)}`;
}

export function formatWhen(value: Date | string | null | undefined) {
  const ms = toMs(value);
  if (ms === null) return "Not started";
  const date = new Date(ms);
  const sameDay = new Date().toDateString() === date.toDateString();
  const time = date.toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
  if (sameDay) return `Today ${time}`;
  return `${date.toLocaleDateString(undefined, { month: "short", day: "numeric" })} ${time}`;
}

/** The workflow asks for something before it can run: inputs it needs, or
 *  an account to connect. */
export function needsSetup(agent: LibraryAgent) {
  const inputs = agent.input_schema as { required?: unknown } | undefined;
  const credentials = agent.credentials_input_schema as
    | { properties?: Record<string, unknown> }
    | undefined;
  const required = Array.isArray(inputs?.required) ? inputs.required : [];
  const accounts = Object.keys(credentials?.properties ?? {});
  return (
    required.length > 0 || accounts.length > 0 || !!agent.trigger_setup_info
  );
}
