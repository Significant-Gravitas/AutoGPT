"use client";

import { ToolUIPart } from "ai";
import { GearIcon } from "@phosphor-icons/react";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";

interface Props {
  part: ToolUIPart;
}

function extractToolName(part: ToolUIPart): string {
  // ToolUIPart.type is "tool-{name}", extract the name portion.
  return part.type.replace(/^tool-/, "");
}

function formatToolName(name: string): string {
  // "search_docs" → "Search docs", "Read" → "Read"
  return name.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

function getAnimationText(part: ToolUIPart): string {
  const label = formatToolName(extractToolName(part));

  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return `Running ${label}…`;
    case "output-available":
      return `${label} completed`;
    case "output-error":
      return `${label} failed`;
    default:
      return `Running ${label}…`;
  }
}

export function GenericTool({ part }: Props) {
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";
  const isError = part.state === "output-error";

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <GearIcon
          size={14}
          weight="regular"
          className={
            isError
              ? "text-red-500"
              : isStreaming
                ? "animate-spin text-neutral-500"
                : "text-neutral-400"
          }
        />
        <MorphingTextAnimation
          text={getAnimationText(part)}
          className={isError ? "text-red-500" : undefined}
        />
      </div>
    </div>
  );
}
