import type { ToolUIPart } from "ai";
import {
  CheckCircleIcon,
  CircleNotchIcon,
  XCircleIcon,
} from "@phosphor-icons/react";

export interface RunBlockInput {
  block_id?: string;
  input_data?: Record<string, unknown>;
}

export interface CredentialsMeta {
  id: string;
  provider: string;
  provider_name?: string;
  type: string;
  types?: string[];
  title: string;
  scopes?: string[];
}

export interface SetupInfo {
  agent_id: string;
  agent_name: string;
  requirements: {
    credentials: CredentialsMeta[];
    inputs: Array<{
      name: string;
      title: string;
      type: string;
      description: string;
      required: boolean;
    }>;
    execution_modes: string[];
  };
  user_readiness: {
    has_all_credentials: boolean;
    missing_credentials: Record<string, CredentialsMeta>;
    ready_to_run: boolean;
  };
}

export interface SetupRequirementsOutput {
  type: "setup_requirements";
  message: string;
  session_id?: string;
  setup_info: SetupInfo;
}

export interface BlockOutput {
  type: "block_output";
  message: string;
  session_id?: string;
  block_id: string;
  block_name: string;
  outputs: Record<string, unknown[]>;
  success: boolean;
}

export interface ErrorOutput {
  type: "error";
  message: string;
  session_id?: string;
  error?: string | null;
  details?: Record<string, unknown> | null;
}

export type RunBlockToolOutput =
  | SetupRequirementsOutput
  | BlockOutput
  | ErrorOutput;

function parseOutput(output: unknown): RunBlockToolOutput | null {
  if (!output) return null;
  if (typeof output === "string") {
    const trimmed = output.trim();
    if (!trimmed) return null;
    try {
      return JSON.parse(trimmed) as RunBlockToolOutput;
    } catch {
      return null;
    }
  }
  if (typeof output === "object") return output as RunBlockToolOutput;
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
      if (output.type === "block_output")
        return `Block ran: ${output.block_name}`;
      if (output.type === "setup_requirements") {
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
