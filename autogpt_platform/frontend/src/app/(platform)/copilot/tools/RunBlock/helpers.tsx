import type { BlockOutputResponse } from "@/app/api/__generated__/models/blockOutputResponse";
import type { ErrorResponse } from "@/app/api/__generated__/models/errorResponse";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import {
  PlayCircleIcon,
  PlayIcon,
  WarningDiamondIcon,
} from "@phosphor-icons/react";
import type { ToolUIPart } from "ai";
import { OrbitLoader } from "../../components/OrbitLoader/OrbitLoader";

/** Block details returned on first run_block attempt (before input_data provided). */
export interface BlockDetailsResponse {
  type: typeof ResponseType.block_details;
  message: string;
  session_id?: string | null;
  block: {
    id: string;
    name: string;
    description: string;
    inputs: Record<string, unknown>;
    outputs: Record<string, unknown>;
    credentials: unknown[];
  };
  user_authenticated: boolean;
}

export interface RunBlockInput {
  block_id?: string;
  block_name?: string;
  input_data?: Record<string, unknown>;
}

export type RunBlockToolOutput =
  | SetupRequirementsResponse
  | BlockDetailsResponse
  | BlockOutputResponse
  | ErrorResponse;

const RUN_BLOCK_OUTPUT_TYPES = new Set<string>([
  ResponseType.setup_requirements,
  ResponseType.block_details,
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

export function isRunBlockDetailsOutput(
  output: RunBlockToolOutput,
): output is BlockDetailsResponse {
  return (
    output.type === ResponseType.block_details ||
    ("block" in output && typeof output.block === "object")
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
    if ("block" in output) return output as BlockDetailsResponse;
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

export function getAnimationText(part: {
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}): string {
  const input = part.input as RunBlockInput | undefined;
  const blockName = input?.block_name?.trim();
  const blockId = input?.block_id?.trim();
  // Prefer block_name if available, otherwise fall back to block_id
  const blockText = blockName
    ? ` "${blockName}"`
    : blockId
      ? ` "${blockId}"`
      : "";

  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return `Running${blockText}`;
    case "output-available": {
      const output = parseOutput(part.output);
      if (!output) return `Running${blockText}`;
      if (isRunBlockBlockOutput(output)) return `Ran "${output.block_name}"`;
      if (isRunBlockDetailsOutput(output))
        return `Details for "${output.block.name}"`;
      if (isRunBlockSetupRequirementsOutput(output)) {
        return `Setup needed for "${output.setup_info.agent_name}"`;
      }
      return "Error running block";
    }
    case "output-error":
      return "Error running block";
    default:
      return "Running the block";
  }
}

export function ToolIcon({
  isStreaming,
  isError,
}: {
  isStreaming?: boolean;
  isError?: boolean;
}) {
  if (isError) {
    return (
      <WarningDiamondIcon size={14} weight="regular" className="text-red-500" />
    );
  }
  if (isStreaming) {
    return <OrbitLoader size={24} />;
  }
  return <PlayIcon size={14} weight="regular" className="text-neutral-400" />;
}

export function AccordionIcon() {
  return <PlayCircleIcon size={32} weight="light" />;
}

export function formatMaybeJson(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function getAccordionMeta(output: RunBlockToolOutput): {
  icon: React.ReactNode;
  title: string;
  titleClassName?: string;
  description?: string;
} {
  const icon = <AccordionIcon />;

  if (isRunBlockBlockOutput(output)) {
    const keys = Object.keys(output.outputs ?? {});
    return {
      icon,
      title: output.block_name,
      description:
        keys.length > 0
          ? `${keys.length} output key${keys.length === 1 ? "" : "s"}`
          : output.message,
    };
  }

  if (isRunBlockDetailsOutput(output)) {
    const inputKeys = Object.keys(
      (output.block.inputs as { properties?: Record<string, unknown> })
        ?.properties ?? {},
    );
    return {
      icon,
      title: output.block.name,
      description:
        inputKeys.length > 0
          ? `${inputKeys.length} input field${inputKeys.length === 1 ? "" : "s"} available`
          : output.message,
    };
  }

  if (isRunBlockSetupRequirementsOutput(output)) {
    const missingCredsCount = Object.keys(
      (output.setup_info.user_readiness?.missing_credentials ?? {}) as Record<
        string,
        unknown
      >,
    ).length;
    return {
      icon,
      title: output.setup_info.agent_name,
      description:
        missingCredsCount > 0
          ? `Missing ${missingCredsCount} credential${missingCredsCount === 1 ? "" : "s"}`
          : output.message,
    };
  }

  return {
    icon: (
      <WarningDiamondIcon size={32} weight="light" className="text-red-500" />
    ),
    title: "Error",
    titleClassName: "text-red-500",
  };
}
