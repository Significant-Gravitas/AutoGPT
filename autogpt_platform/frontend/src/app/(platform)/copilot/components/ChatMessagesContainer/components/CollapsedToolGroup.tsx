"use client";

import { useId, useState } from "react";
import type { ToolUIPart } from "ai";
import {
  type ToolCategory,
  extractToolName,
  getAnimationText,
  getToolCategory,
} from "../../../tools/GenericTool/helpers";
import {
  AlertDiamondIcon,
  ArrowRight01Icon,
  CheckListIcon,
  CheckmarkCircle02Icon,
  ComputerIcon,
  Delete02Icon,
  FileEmpty02Icon,
  Files01Icon,
  Globe02Icon,
  PencilIcon,
  ReloadIcon,
  Search01Icon,
  Settings01Icon,
  TerminalIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
    return <Icon icon={AlertDiamondIcon} size={14} className="text-red-500" />;
  }

  const iconClass = "text-green-500";
  switch (category) {
    case "bash":
      return <Icon icon={TerminalIcon} size={14} className={iconClass} />;
    case "web":
      return <Icon icon={Globe02Icon} size={14} className={iconClass} />;
    case "browser":
      return <Icon icon={ComputerIcon} size={14} className={iconClass} />;
    case "file-read":
    case "file-write":
      return <Icon icon={FileEmpty02Icon} size={14} className={iconClass} />;
    case "file-delete":
      return <Icon icon={Delete02Icon} size={14} className={iconClass} />;
    case "file-list":
      return <Icon icon={Files01Icon} size={14} className={iconClass} />;
    case "search":
      return <Icon icon={Search01Icon} size={14} className={iconClass} />;
    case "edit":
      return <Icon icon={PencilIcon} size={14} className={iconClass} />;
    case "todo":
      return <Icon icon={CheckListIcon} size={14} className={iconClass} />;
    case "compaction":
      return <Icon icon={ReloadIcon} size={14} className={iconClass} />;
    default:
      return <Icon icon={Settings01Icon} size={14} className={iconClass} />;
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
    <div className="py-1 text-xs opacity-50 transition-opacity group-hover:opacity-100">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        aria-controls={panelId}
        className="flex items-center gap-1.5 !text-xs text-muted-foreground transition-colors hover:text-foreground"
      >
        <Icon
          icon={ArrowRight01Icon}
          size={12}
          className={
            "transition-transform duration-150 " + (expanded ? "rotate-90" : "")
          }
        />
        {errorCount > 0 ? (
          <Icon icon={AlertDiamondIcon} size={14} className="text-red-500" />
        ) : (
          <Icon
            icon={CheckmarkCircle02Icon}
            size={14}
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
