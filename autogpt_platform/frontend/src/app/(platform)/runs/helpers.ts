import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import {
  formatCost,
  formatWhen,
  runElapsedMs,
  statusView,
  triggerLabel,
} from "@/app/(platform)/copilot/components/WorkArea/WorkflowRunTab/runs";
import {
  toMs,
  type Trace,
  type TraceStep,
} from "@/app/(platform)/copilot/components/WorkArea/WorkflowRunTab/trace";

import type { AgentRun, RunStep, StepKind } from "./affect/agent-run-detail";

const STATUS_KEYS = {
  live: "running",
  ok: "done",
  failed: "failed",
  idle: "idle",
  waiting: "idle",
} as const;

/** A run that answers 4xx (gone, never a run, a malformed id) will not
 *  appear on a retry: stop at once. A server error gets one quick retry. */
export function retryRunQuery(failureCount: number, error: unknown): boolean {
  if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
    return false;
  }
  return failureCount < 1;
}

export const RUN_RETRY_DELAY_MS = 400;

/** Whether the run could not be read because of the server, not the id. */
export function isServerFailure(error: unknown): boolean {
  return !(error instanceof ApiError) || error.status >= 500;
}

/** The page for one run, beside the chat or on its own. */
export function runHref(graphID: string, runID: string): string {
  return `/runs/${encodeURIComponent(graphID)}/${encodeURIComponent(runID)}`;
}

export function stepKind(step: TraceStep): StepKind {
  if (step.runsWorkflow) return "agent";
  if (step.kind === "ai") return "llm";
  if (step.kind === "integration") return "tool";
  if (step.kind === "io") return "io";
  if (step.kind === "logic") return "logic";
  if (step.kind === "data") return "data";
  return "block";
}

/** What a step's values copy as. */
export function asText(value: Record<string, unknown> | null): string {
  if (!value || Object.keys(value).length === 0) return "";
  return JSON.stringify(value, null, 2);
}

export function withoutError(
  output: TraceStep["output"],
): Record<string, unknown[]> | null {
  if (!output) return null;
  const entries = Object.entries(output).filter(([key]) => key !== "error");
  return entries.length ? Object.fromEntries(entries) : null;
}

/** A block run as one of the block's steps; blocks not reached yet are left
 *  out (the block has no waiting row). */
export function toRunStep(step: TraceStep): RunStep {
  const running = step.state === "running";
  return {
    id: step.id,
    label: step.name,
    kind: stepKind(step),
    agent: step.runsWorkflow ? { id: step.name, color: "" } : undefined,
    start: step.offsetMs / 1000,
    duration: running ? undefined : step.durationMs / 1000,
    failed: step.state === "failed",
    error: step.error ?? undefined,
    input: asText(step.input),
    output: asText(withoutError(step.output)),
  };
}

interface RunArgs {
  graphID: string;
  title: string;
  run: GraphExecution;
  trace: Trace;
  isLive: boolean;
  now: number;
}

/** The run as the block reads it, and how far it has come while it runs. */
export function toAgentRun({
  graphID,
  title,
  run,
  trace,
  isLive,
  now,
}: RunArgs) {
  const started = toMs(run.started_at) ?? now;
  const elapsedMs = runElapsedMs(run, now) ?? trace.totalMs;
  const view = statusView(run.status);
  const agentRun: AgentRun = {
    id: run.id,
    title,
    agent: { id: graphID, name: title, color: "" },
    model: "",
    startedAt: started,
    total: isLive ? undefined : Math.max(elapsedMs, trace.totalMs) / 1000,
    subtitle: `${triggerLabel(run)} · ${formatWhen(run.started_at)}`,
    status: { key: STATUS_KEYS[view.tone], label: view.label },
    costText: formatCost(run.stats?.cost),
    steps: trace.steps
      .filter((step) => step.state !== "waiting")
      .map(toRunStep),
  };
  return {
    agentRun,
    elapsed: isLive ? Math.max(0, now - started) / 1000 : undefined,
  };
}
