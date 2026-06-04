"use client";

import { useId, useState } from "react";
import {
  ArrowsClockwiseIcon,
  CaretRightIcon,
  CheckCircleIcon,
  FileIcon,
  FilesIcon,
  GearIcon,
  GlobeIcon,
  ListChecksIcon,
  MagnifyingGlassIcon,
  MonitorIcon,
  PencilSimpleIcon,
  TerminalIcon,
  TrashIcon,
  WarningDiamondIcon,
} from "@phosphor-icons/react";
import type { ToolUIPart } from "ai";
import {
  type ToolCategory,
  extractToolName,
  getAnimationText,
  getToolCategory,
} from "../../../tools/GenericTool/helpers";

interface Props {
  parts: ToolUIPart[];
}

/** Category icon matching GenericTool's ToolIcon for completed states. */
function EntryIcon({
  category,
  isError,
}: {
  category: ToolCategory;
  isError: boolean;
}) {
  if (isError) {
    return (
      <WarningDiamondIcon size={14} weight="regular" className="text-red-500" />
    );
  }

  const iconClass = "text-green-500";
  switch (category) {
    case "bash":
      return <TerminalIcon size={14} weight="regular" className={iconClass} />;
    case "web":
      return <GlobeIcon size={14} weight="regular" className={iconClass} />;
    case "browser":
      return <MonitorIcon size={14} weight="regular" className={iconClass} />;
    case "file-read":
    case "file-write":
      return <FileIcon size={14} weight="regular" className={iconClass} />;
    case "file-delete":
      return <TrashIcon size={14} weight="regular" className={iconClass} />;
    case "file-list":
      return <FilesIcon size={14} weight="regular" className={iconClass} />;
    case "search":
      return (
        <MagnifyingGlassIcon size={14} weight="regular" className={iconClass} />
      );
    case "edit":
      return (
        <PencilSimpleIcon size={14} weight="regular" className={iconClass} />
      );
    case "todo":
      return (
        <ListChecksIcon size={14} weight="regular" className={iconClass} />
      );
    case "compaction":
      return (
        <ArrowsClockwiseIcon size={14} weight="regular" className={iconClass} />
      );
    default:
      return <GearIcon size={14} weight="regular" className={iconClass} />;
  }
}

export function CollapsedToolGroup({ parts }: Props) {
  const [expanded, setExpanded] = useState(false);
  const panelId = useId();

  const errorCount = parts.filter((p) => p.state === "output-error").length;
  const label =
    errorCount > 0
      ? `${parts.length} tool calls (${errorCount} failed)`
      : `${parts.length} tool calls completed`;

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
          className="ml-5 mt-1 space-y-0.5 border-l border-neutral-200 pl-3"
        >
          {parts.map((part) => {
            const toolName = extractToolName(part);
            const category = getToolCategory(toolName);
            const text = getAnimationText(part, category);
            const isError = part.state === "output-error";

            return (
              <div
                key={part.toolCallId}
                className={
                  "flex items-center gap-1.5 text-xs " +
                  (isError ? "text-red-500" : "text-muted-foreground")
                }
              >
                <EntryIcon category={category} isError={isError} />
                <span>{text}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
