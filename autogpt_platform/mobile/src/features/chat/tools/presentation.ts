import type { SFSymbol } from "expo-symbols";

import { asRecord, nonEmptyString, truncate } from "../../../lib/json";
import { familyForResponseType, type CardFamily } from "./catalog";

/**
 * A tool call as the timeline knows it. `output` is undefined while no
 * result exists; `state` says whether one is still coming.
 */
export interface ToolPart {
  toolCallID: string;
  toolName: string;
  displayName: string | null;
  input: unknown;
  output: unknown;
  state: "running" | "done" | "failed" | "interrupted" | "unrecorded";
}

export interface ToolPresentation {
  title: string;
  statusLabel: string;
  symbol: SFSymbol;
  /** The output's `type` discriminator, when it has one. */
  responseType: string | null;
  family: CardFamily | null;
  /** Whether the output is a type this version of the app knows about. */
  recognized: boolean;
  summary: string | null;
  coverageNote: string;
}

const SUMMARY_FIELDS = [
  "message",
  "error",
  "summary",
  "text",
  "result",
  "content",
  "title",
];

export function presentTool(part: ToolPart): ToolPresentation {
  const output = asRecord(part.output);
  const responseType = nonEmptyString(output?.type);
  const family = responseType ? familyForResponseType(responseType) : null;
  const failed =
    part.state === "failed" ||
    responseType === "error" ||
    responseType === "input_validation_error" ||
    output?.success === false;
  const state = failed && part.state === "done" ? "failed" : part.state;
  return {
    title: part.displayName ?? humanizeToolName(part.toolName),
    statusLabel: STATUS_LABELS[state],
    symbol:
      state === "failed"
        ? "exclamationmark.triangle"
        : (family?.symbol ?? "wrench.and.screwdriver"),
    responseType,
    family,
    recognized: family !== null,
    summary: summarize(part.output),
    coverageNote: coverageNote(part, responseType, family),
  };
}

const STATUS_LABELS: Record<ToolPart["state"], string> = {
  running: "Working…",
  done: "Done",
  failed: "Failed",
  interrupted: "Interrupted",
  unrecorded: "No result recorded",
};

function coverageNote(
  part: ToolPart,
  responseType: string | null,
  family: CardFamily | null,
): string {
  if (part.state === "running") {
    return "Still running. The result appears here when it finishes.";
  }
  if (part.output === undefined) {
    return "No result was saved for this step.";
  }
  if (!responseType) {
    return "This step returned plain output. It's shown below as text.";
  }
  if (!family) {
    return "This result type isn't recognized by this version of the app, so only its raw details are shown and none of its actions are offered.";
  }
  return family.actionable
    ? `The native ${family.label.toLowerCase()} card isn't built yet, so its actions aren't available in the app. Its details are shown read-only.`
    : `The native ${family.label.toLowerCase()} card isn't built yet. Its details are shown read-only.`;
}

/** `run_agent` → "Run agent"; `mcp__server__tool` → "Tool (server)". */
export function humanizeToolName(name: string): string {
  if (!name) return "Tool";
  if (name.startsWith("mcp__")) {
    const [, server, ...rest] = name.split("__");
    const tool = rest.join(" ") || server;
    return `${sentenceCase(tool)}${rest.length && server ? ` (${server})` : ""}`;
  }
  return sentenceCase(name);
}

function sentenceCase(raw: string): string {
  const spaced = raw
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .trim()
    .toLowerCase();
  return spaced ? spaced.charAt(0).toUpperCase() + spaced.slice(1) : "Tool";
}

function summarize(output: unknown): string | null {
  if (typeof output === "string") {
    const trimmed = output.trim();
    if (!trimmed) return null;
    if (/^[[{]/.test(trimmed)) return "The result couldn't be read.";
    return truncate(trimmed.replace(/\s+/g, " "), 240);
  }
  const record = asRecord(output);
  if (!record) return null;
  for (const field of SUMMARY_FIELDS) {
    const value = nonEmptyString(record[field]);
    if (value) return truncate(value.replace(/\s+/g, " "), 240);
  }
  return null;
}
