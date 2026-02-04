import { ToolUIPart } from "ai";
import type { BlockListResponse } from "@/app/api/__generated__/models/blockListResponse";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import { FindBlockInput, FindBlockToolPart } from "./FindBlocks";
import {
  CheckCircleIcon,
  CircleNotchIcon,
  XCircleIcon,
} from "@phosphor-icons/react";

function parseOutput(output: unknown): BlockListResponse | null {
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
    if (type === ResponseType.block_list || "blocks" in output) {
      return output as BlockListResponse;
    }
  }
  return null;
}

export function getAnimationText(part: FindBlockToolPart): string {
  switch (part.state) {
    case "input-streaming":
      return "Searching blocks for you";

    case "input-available": {
      const query = (part.input as FindBlockInput).query;
      return `Finding "${query}" blocks`;
    }

    case "output-available": {
      const parsed = parseOutput(part.output);
      if (parsed) {
        return `Found ${parsed.count} "${(part.input as FindBlockInput).query}" blocks`;
      }
      return "Found blocks";
    }

    case "output-error":
      return "Error finding blocks";

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
