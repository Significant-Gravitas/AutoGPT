import type { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import type { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";
import type { NodeModel } from "@/app/api/__generated__/models/nodeModel";
import { beautifyString } from "@/lib/utils";

/** The families a block's colour comes from. */
export type BlockKind =
  | "ai"
  | "integration"
  | "logic"
  | "io"
  | "data"
  | "other";
export type StepState = "waiting" | "running" | "done" | "failed" | "stopped";

export interface TraceStep {
  id: string;
  name: string;
  kind: BlockKind;
  state: StepState;
  /** Milliseconds from the run's start to the step's start. */
  offsetMs: number;
  /** Zero until the step starts; grows with the clock while it runs. */
  durationMs: number;
  input: Record<string, unknown> | null;
  output: Record<string, unknown[]> | null;
  error: string | null;
  /** The block runs another workflow (an agent block). */
  runsWorkflow?: boolean;
}

export interface Trace {
  steps: TraceStep[];
  /** The axis length: the run so far, never under a second. */
  totalMs: number;
}

const MIN_AXIS_MS = 1000;
const AI_CATEGORIES = new Set(["AI", "SAFETY"]);
const LOGIC_CATEGORIES = new Set(["LOGIC", "AGENT"]);
const LOCAL_CATEGORIES = new Set(["TEXT", "DATA", "BASIC"]);

export function toMs(value: Date | string | null | undefined): number | null {
  if (!value) return null;
  const ms = value instanceof Date ? value.getTime() : Date.parse(value);
  return Number.isNaN(ms) ? null : ms;
}

export function kindOf(block: BlockInfo | undefined): BlockKind {
  if (!block) return "other";
  const categories = block.categories.map((entry) => entry.category);
  const ui = block.uiType.toLowerCase();
  // Only the workflow's own input and output blocks: a block filed under
  // OUTPUT (a web request, a notification) reaches outside, so it reads as
  // an integration below.
  if (ui === "input" || ui === "output") return "io";
  if (categories.some((c) => AI_CATEGORIES.has(c))) return "ai";
  if (categories.some((c) => LOGIC_CATEGORIES.has(c))) return "logic";
  if (categories.length > 0 && categories.every((c) => LOCAL_CATEGORIES.has(c)))
    return "data";
  return categories.length > 0 ? "integration" : "other";
}

function runsWorkflow(block: BlockInfo | undefined): boolean {
  return !!block?.categories.some((entry) => entry.category === "AGENT");
}

function stateOf(status: string): StepState {
  switch (status) {
    case "RUNNING":
      return "running";
    case "COMPLETED":
      return "done";
    case "FAILED":
      return "failed";
    case "TERMINATED":
      return "stopped";
    default:
      return "waiting";
  }
}

function nameOf(block: BlockInfo | undefined, node: NodeModel | undefined) {
  const custom = node?.metadata?.customized_name;
  if (typeof custom === "string" && custom.trim()) return custom.trim();
  if (!block) return "Block";
  return beautifyString(block.name)
    .replace(/ Block$/, "")
    .trim();
}

function errorOf(execution: NodeExecutionResult) {
  const error = execution.output_data?.error?.[0];
  if (error == null) return null;
  return typeof error === "string" ? error : JSON.stringify(error);
}

interface BuildArgs {
  executions: NodeExecutionResult[];
  nodes: NodeModel[];
  blocks: BlockInfo[];
  runStartedAt: Date | string | null | undefined;
  runEndedAt: Date | string | null | undefined;
  isLive: boolean;
  now: number;
}

/** The run's block runs as rows on one time axis, in the order they started;
 *  rows not started yet come last. While the run is live the blocks it has
 *  not reached yet wait at the bottom. */
export function buildTrace(args: BuildArgs): Trace {
  const { executions, nodes, blocks, isLive, now } = args;
  const blockByID = new Map(blocks.map((block) => [block.id, block]));
  const nodeByID = new Map(nodes.map((node) => [node.id, node]));
  const starts = executions
    .map((execution) => toMs(execution.start_time))
    .filter((ms): ms is number => ms !== null);
  const origin =
    toMs(args.runStartedAt) ?? (starts.length ? Math.min(...starts) : now);

  const ran = executions.map((execution) => {
    const block = blockByID.get(execution.block_id);
    const start = toMs(execution.start_time);
    const end = toMs(execution.end_time);
    const state = stateOf(execution.status);
    const finish = end ?? (state === "running" && isLive ? now : start);
    const step: TraceStep = {
      id: execution.node_exec_id,
      name: nameOf(block, nodeByID.get(execution.node_id)),
      kind: kindOf(block),
      runsWorkflow: runsWorkflow(block),
      state,
      offsetMs:
        start === null
          ? Math.max(0, (isLive ? now : origin) - origin)
          : Math.max(0, start - origin),
      durationMs:
        start === null || finish === null ? 0 : Math.max(0, finish - start),
      input: execution.input_data ?? null,
      output: execution.output_data ?? null,
      error: errorOf(execution),
    };
    return { step, start, added: toMs(execution.add_time) ?? now };
  });
  // Started rows by start time; the queued ones after them, as queued.
  ran.sort(
    (a, b) =>
      (a.start ?? Number.MAX_SAFE_INTEGER) -
        (b.start ?? Number.MAX_SAFE_INTEGER) || a.added - b.added,
  );

  const reached = new Set(executions.map((execution) => execution.node_id));
  const ahead: TraceStep[] = isLive
    ? nodes
        .filter((node) => node.id && !reached.has(node.id))
        .map((node) => {
          const block = blockByID.get(node.block_id);
          return {
            id: `node:${node.id}`,
            name: nameOf(block, node),
            kind: kindOf(block),
            state: "waiting" as const,
            offsetMs: Math.max(0, now - origin),
            durationMs: 0,
            input: null,
            output: null,
            error: null,
          };
        })
    : [];

  const steps = [...ran.map((entry) => entry.step), ...ahead];
  const lastEnd = Math.max(
    0,
    ...steps.map((step) => step.offsetMs + step.durationMs),
  );
  const ended = toMs(args.runEndedAt);
  const liveEnd = isLive ? now - origin : ended !== null ? ended - origin : 0;
  return { steps, totalMs: Math.max(MIN_AXIS_MS, lastEnd, liveEnd) };
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const seconds = ms / 1000;
  if (seconds < 10) return `${seconds.toFixed(1)}s`;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  if (minutes < 60) return `${minutes}m ${String(rest).padStart(2, "0")}s`;
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, "0")}m`;
}
