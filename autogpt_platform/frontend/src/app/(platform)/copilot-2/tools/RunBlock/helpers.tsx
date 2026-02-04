import type { ToolUIPart } from "ai";
import {
  CheckCircleIcon,
  CircleNotchIcon,
  XCircleIcon,
} from "@phosphor-icons/react";
import type { BlockOutputResponse } from "@/app/api/__generated__/models/blockOutputResponse";
import type { ErrorResponse } from "@/app/api/__generated__/models/errorResponse";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";

export interface RunBlockInput {
  block_id?: string;
  input_data?: Record<string, unknown>;
}

export type RunBlockToolOutput =
  | SetupRequirementsResponse
  | BlockOutputResponse
  | ErrorResponse;

const RUN_BLOCK_OUTPUT_TYPES = new Set<string>([
  ResponseType.setup_requirements,
  ResponseType.block_output,
  ResponseType.error,
]);

export function isRunBlockSetupRequirementsOutput(
  output: RunBlockToolOutput,
): output is SetupRequirementsResponse {
  return (
    output.type === ResponseType.setup_requirements ||
    ("setup_info" in output && typeof output.setup_info === "object")
  );
}

export function isRunBlockBlockOutput(
  output: RunBlockToolOutput,
): output is BlockOutputResponse {
  return output.type === ResponseType.block_output || "block_id" in output;
}

export function isRunBlockErrorOutput(
  output: RunBlockToolOutput,
): output is ErrorResponse {
  return output.type === ResponseType.error || "error" in output;
}

function parseOutput(output: unknown): RunBlockToolOutput | null {
  if (!output) return null;
  if (typeof output === "string") {
    const trimmed = output.trim();
    if (!trimmed) return null;
    try {
      return parseOutput(JSON.parse(trimmed) as unknown);
    } catch {
      return null;
    }
  }
  if (typeof output === "object") {
    const type = (output as { type?: unknown }).type;
    if (typeof type === "string" && RUN_BLOCK_OUTPUT_TYPES.has(type)) {
      return output as RunBlockToolOutput;
    }
    if ("block_id" in output) return output as BlockOutputResponse;
    if ("setup_info" in output) return output as SetupRequirementsResponse;
    if ("error" in output || "details" in output)
      return output as ErrorResponse;
  }
  return null;
}

export function getRunBlockToolOutput(
  part: unknown,
): RunBlockToolOutput | null {
  if (!part || typeof part !== "object") return null;
  return parseOutput((part as { output?: unknown }).output);
}

function getBlockLabel(input: RunBlockInput | undefined): string | null {
  const blockId = input?.block_id?.trim();
  if (!blockId) return null;
  return `Block ${blockId.slice(0, 8)}…`;
}

export function getAnimationText(part: {
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}): string {
  const input = part.input as RunBlockInput | undefined;
  const blockLabel = getBlockLabel(input);

  switch (part.state) {
    case "input-streaming":
      return "Preparing to run block";
    case "input-available":
      return blockLabel ? `Running ${blockLabel}` : "Running block";
    case "output-available": {
      const output = parseOutput(part.output);
      if (!output) return "Block run updated";
      if (isRunBlockBlockOutput(output))
        return `Block ran: ${output.block_name}`;
      if (isRunBlockSetupRequirementsOutput(output)) {
        return `Needs setup: ${output.setup_info.agent_name}`;
      }
      return "Error running block";
    }
    case "output-error":
      return "Error running block";
    default:
      return "Processing";
  }
}

export function StateIcon({ state }: { state: ToolUIPart["state"] }) {
  switch (state) {
    case "input-streaming":
    case "input-available":
      return (
        <CircleNotchIcon
          className="h-4 w-4 animate-spin text-muted-foreground"
          weight="bold"
        />
      );
    case "output-available":
      return <CheckCircleIcon className="h-4 w-4 text-green-500" />;
    case "output-error":
      return <XCircleIcon className="h-4 w-4 text-red-500" />;
    default:
      return null;
  }
}

export function formatMaybeJson(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
