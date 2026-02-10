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

export function getAnimationText(part: {
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}): string {
  const input = part.input as RunBlockInput | undefined;
  const blockId = input?.block_id?.trim();
  const blockText = blockId ? ` "${blockId}"` : "";

  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return `Running the block${blockText}`;
    case "output-available": {
      const output = parseOutput(part.output);
      if (!output) return `Running the block${blockText}`;
      if (isRunBlockBlockOutput(output)) return `Ran "${output.block_name}"`;
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
      icon: <OrbitLoader size={24} className="text-neutral-700" />,
      title: output.block_name,
      description:
        keys.length > 0
          ? `${keys.length} output key${keys.length === 1 ? "" : "s"}`
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
