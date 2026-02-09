import type { BlockListResponse } from "@/app/api/__generated__/models/blockListResponse";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import { CubeIcon, PackageIcon } from "@phosphor-icons/react";
import { FindBlockInput, FindBlockToolPart } from "./FindBlocks";

export function parseOutput(output: unknown): BlockListResponse | null {
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
  const query = (part.input as FindBlockInput | undefined)?.query?.trim();
  const queryText = query ? ` matching "${query}"` : "";

  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return `Searching for blocks${queryText}`;

    case "output-available": {
      const parsed = parseOutput(part.output);
      if (parsed) {
        return `Found ${parsed.count} block${parsed.count === 1 ? "" : "s"}${queryText}`;
      }
      return `Searching for blocks${queryText}`;
    }

    case "output-error":
      return `Error finding blocks${queryText}`;

    default:
      return "Searching for blocks";
  }
}

export function ToolIcon({
  isStreaming,
  isError,
}: {
  isStreaming?: boolean;
  isError?: boolean;
}) {
  return (
    <PackageIcon
      size={14}
      weight="regular"
      className={
        isError
          ? "text-red-500"
          : isStreaming
            ? "text-neutral-500"
            : "text-neutral-400"
      }
    />
  );
}

export function AccordionIcon() {
  return <CubeIcon size={32} weight="light" />;
}
