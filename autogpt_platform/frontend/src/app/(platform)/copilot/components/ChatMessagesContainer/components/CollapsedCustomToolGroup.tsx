"use client";

import { useId, useState } from "react";
import {
  CaretRightIcon,
  CheckCircleIcon,
  WarningDiamondIcon,
} from "@phosphor-icons/react";
import type { ToolUIPart } from "ai";
import { MessagePartRenderer } from "./MessagePartRenderer";

const TOOL_LABELS: Record<string, string> = {
  "tool-find_agent": "agent searches",
  "tool-find_library_agent": "library searches",
  "tool-find_block": "block searches",
  "tool-search_docs": "doc searches",
  "tool-get_doc_page": "doc lookups",
  "tool-search_feature_requests": "feature request searches",
};

function getGroupLabel(toolType: string, count: number): string {
  const label = TOOL_LABELS[toolType];
  if (label) return `${count} ${label}`;
  const name = toolType.replace(/^tool-/, "").replace(/_/g, " ");
  return `${count} ${name} calls`;
}

interface Props {
  toolType: string;
  parts: ToolUIPart[];
  messageID: string;
}

export function CollapsedCustomToolGroup({
  toolType,
  parts,
  messageID,
}: Props) {
  const [expanded, setExpanded] = useState(false);
  const panelId = useId();

  const errorCount = parts.filter((p) => p.state === "output-error").length;
  const label =
    errorCount > 0
      ? `${getGroupLabel(toolType, parts.length)} (${errorCount} failed)`
      : getGroupLabel(toolType, parts.length);

  return (
    <div className="py-1">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        aria-controls={panelId}
        className="flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
      >
        <CaretRightIcon
          size={12}
          weight="bold"
          className={
            "transition-transform duration-150 " + (expanded ? "rotate-90" : "")
          }
        />
        {errorCount > 0 ? (
          <WarningDiamondIcon
            size={14}
            weight="regular"
            className="text-red-500"
          />
        ) : (
          <CheckCircleIcon
            size={14}
            weight="regular"
            className="text-green-500"
          />
        )}
        <span>{label}</span>
      </button>

      {expanded && (
        <div
          id={panelId}
          className="ml-5 mt-1 border-l border-neutral-200 pl-3"
        >
          {parts.map((part, i) => (
            <MessagePartRenderer
              key={part.toolCallId}
              part={part}
              messageID={messageID}
              partIndex={i}
            />
          ))}
        </div>
      )}
    </div>
  );
}
