"use client";

import { cn } from "@/lib/utils";
import {
  CaretDownIcon,
  CheckCircleIcon,
  FileIcon,
  FilesIcon,
  GearIcon,
  GlobeIcon,
  LightningIcon,
  ListChecksIcon,
  MagnifyingGlassIcon,
  PencilSimpleIcon,
  PlusCircleIcon,
  RocketLaunchIcon,
  TerminalIcon,
  WarningDiamondIcon,
} from "@phosphor-icons/react";
import { UIDataTypes, UIMessage, UITools } from "ai";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import { OrbitLoader } from "../../../../components/OrbitLoader/OrbitLoader";
import { MessagePartRenderer } from "./MessagePartRenderer";

type Part = UIMessage<unknown, UIDataTypes, UITools>["parts"][number];

interface ToolPart {
  type: string;
  state: string;
  input?: unknown;
  output?: unknown;
}

interface Props {
  toolType: string;
  parts: Part[];
  partIndices: number[];
  messageID: string;
}

/* ------------------------------------------------------------------ */
/*  Group icon                                                         */
/* ------------------------------------------------------------------ */

function GroupIcon({ toolType }: { toolType: string }) {
  const name = toolType.replace(/^tool-/, "");
  const cls = "text-neutral-500";
  const size = 16;

  switch (name) {
    case "find_agent":
    case "find_library_agent":
    case "find_block":
    case "Grep":
    case "search_docs":
    case "search_feature_requests":
      return (
        <MagnifyingGlassIcon size={size} weight="regular" className={cls} />
      );
    case "run_block":
      return <LightningIcon size={size} weight="regular" className={cls} />;
    case "run_agent":
    case "schedule_agent":
      return <RocketLaunchIcon size={size} weight="regular" className={cls} />;
    case "create_agent":
    case "create_feature_request":
      return <PlusCircleIcon size={size} weight="regular" className={cls} />;
    case "edit_agent":
    case "Edit":
      return <PencilSimpleIcon size={size} weight="regular" className={cls} />;
    case "bash_exec":
      return <TerminalIcon size={size} weight="regular" className={cls} />;
    case "Read":
    case "read_workspace_file":
    case "Write":
    case "write_workspace_file":
      return <FileIcon size={size} weight="regular" className={cls} />;
    case "WebSearch":
    case "WebFetch":
    case "web_fetch":
      return <GlobeIcon size={size} weight="regular" className={cls} />;
    case "Glob":
    case "list_workspace_files":
      return <FilesIcon size={size} weight="regular" className={cls} />;
    case "get_doc_page":
    case "view_agent_output":
      return <FileIcon size={size} weight="regular" className={cls} />;
    case "TodoWrite":
      return <ListChecksIcon size={size} weight="regular" className={cls} />;
    default:
      return <GearIcon size={size} weight="regular" className={cls} />;
  }
}

/* ------------------------------------------------------------------ */
/*  Group label                                                        */
/* ------------------------------------------------------------------ */

function getGroupLabel(toolType: string, count: number): string {
  const name = toolType.replace(/^tool-/, "");

  switch (name) {
    case "find_agent":
    case "find_library_agent":
      return `Searched for agents — ${count} searches`;
    case "find_block":
      return `Searched for blocks — ${count} searches`;
    case "search_docs":
    case "get_doc_page":
      return `Searched documentation — ${count} lookups`;
    case "run_block":
      return `Executed blocks — ${count} runs`;
    case "run_agent":
      return `Ran agents — ${count} runs`;
    case "schedule_agent":
      return `Scheduled agents — ${count} schedules`;
    case "create_agent":
      return `Created agents — ${count} agents`;
    case "edit_agent":
      return `Edited agents — ${count} edits`;
    case "view_agent_output":
      return `Viewed outputs — ${count} views`;
    case "search_feature_requests":
      return `Searched feature requests — ${count} searches`;
    case "create_feature_request":
      return `Created feature requests — ${count} requests`;
    case "bash_exec":
      return `Ran commands — ${count} commands`;
    case "Read":
    case "read_workspace_file":
      return `Read files — ${count} files`;
    case "Write":
    case "write_workspace_file":
      return `Wrote files — ${count} files`;
    case "WebSearch":
      return `Searched the web — ${count} searches`;
    case "WebFetch":
    case "web_fetch":
      return `Fetched web pages — ${count} pages`;
    case "Glob":
    case "list_workspace_files":
      return `Listed files — ${count} searches`;
    case "Grep":
      return `Searched in files — ${count} searches`;
    case "Edit":
      return `Edited files — ${count} edits`;
    case "TodoWrite":
      return `Updated tasks — ${count} updates`;
    default: {
      const formatted = name.replace(/_/g, " ");
      return `${formatted.charAt(0).toUpperCase() + formatted.slice(1)} — ${count} calls`;
    }
  }
}

/* ------------------------------------------------------------------ */
/*  Sub-row helpers                                                    */
/* ------------------------------------------------------------------ */

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trimEnd() + "…";
}

function getPartSummary(part: Part): string {
  const tp = part as ToolPart;
  const input = tp.input;

  if (input && typeof input === "object") {
    const inp = input as Record<string, unknown>;
    const fields = [
      inp.query,
      inp.name,
      inp.file_path,
      inp.path,
      inp.command,
      inp.url,
      inp.pattern,
      inp.agent_id,
      inp.block_id,
    ];
    for (const f of fields) {
      if (typeof f === "string" && f.trim()) {
        return truncate(f.trim(), 60);
      }
    }
  }

  const toolName = part.type.replace(/^tool-/, "").replace(/_/g, " ");
  return toolName.charAt(0).toUpperCase() + toolName.slice(1);
}

function SubRowStateIcon({ part }: { part: Part }) {
  const tp = part as ToolPart;

  if (tp.state === "input-streaming" || tp.state === "input-available") {
    return <OrbitLoader size={12} />;
  }
  if (tp.state === "output-error") {
    return (
      <WarningDiamondIcon size={12} weight="regular" className="text-red-500" />
    );
  }
  if (tp.state === "output-available") {
    return (
      <CheckCircleIcon size={12} weight="regular" className="text-green-500" />
    );
  }
  return <OrbitLoader size={12} />;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function ToolGroupRow({
  toolType,
  parts,
  partIndices,
  messageID,
}: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandedSubRows, setExpandedSubRows] = useState<Set<number>>(
    () => new Set(),
  );
  const shouldReduceMotion = useReducedMotion();

  const hasStreamingParts = parts.some((p) => {
    const tp = p as ToolPart;
    return tp.state === "input-streaming" || tp.state === "input-available";
  });

  function toggleSubRow(index: number) {
    setExpandedSubRows((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }

  const springTransition = shouldReduceMotion
    ? { duration: 0 }
    : { type: "spring" as const, bounce: 0.25, duration: 0.4 };

  return (
    <div className="py-2">
      <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
        {/* Group summary header */}
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex w-full items-center gap-2 text-left text-sm"
        >
          <span className="flex shrink-0 items-center">
            {hasStreamingParts ? (
              <OrbitLoader size={16} />
            ) : (
              <GroupIcon toolType={toolType} />
            )}
          </span>
          <span className="flex-1 font-medium text-gray-700">
            {getGroupLabel(toolType, parts.length)}
          </span>
          <CaretDownIcon
            className={cn(
              "h-4 w-4 shrink-0 text-slate-400 transition-transform",
              isExpanded && "rotate-180",
            )}
            weight="bold"
          />
        </button>

        {/* Expanded sub-rows */}
        <AnimatePresence initial={false}>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={springTransition}
              className="overflow-hidden"
            >
              <div className="mt-2 space-y-0.5 border-t border-slate-200 pt-2">
                {parts.map((part, i) => {
                  const isSubExpanded = expandedSubRows.has(i);
                  return (
                    <div key={partIndices[i]}>
                      {/* Compact sub-row */}
                      <button
                        type="button"
                        onClick={() => toggleSubRow(i)}
                        className="flex w-full items-center gap-2 rounded px-1 py-1 text-left text-xs hover:bg-slate-100"
                      >
                        <SubRowStateIcon part={part} />
                        <span className="flex-1 truncate text-gray-600">
                          {getPartSummary(part)}
                        </span>
                        <CaretDownIcon
                          className={cn(
                            "h-3 w-3 shrink-0 text-slate-400 transition-transform",
                            isSubExpanded && "rotate-180",
                          )}
                          weight="bold"
                        />
                      </button>

                      {/* Full output (deep expand) */}
                      <AnimatePresence initial={false}>
                        {isSubExpanded && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={springTransition}
                            className="overflow-hidden pl-5"
                          >
                            <MessagePartRenderer
                              part={part}
                              messageID={messageID}
                              partIndex={partIndices[i]}
                            />
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  );
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
