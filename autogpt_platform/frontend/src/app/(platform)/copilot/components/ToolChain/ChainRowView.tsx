"use client";

import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { useEffect, useRef, useState } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { useCopilotUIStore } from "../../store";
import { ACCORDION_PANEL, accordionState, PANEL_REVEAL } from "./accordion";
import { EXPERT_CHANGE_TOOLS } from "./ExpertCards";
import type { ChainRow } from "./helpers";
import { ProviderIcon, RowIcon } from "./RowIcon";
import { useSubSessionEffectiveStatus } from "./SubSessionLive";
import { SwapText } from "./SwapText";
import { getCatalogLabel } from "./toolCatalog";
import { ToolResult } from "./ToolResult";
import { ToolStatusBadge } from "./ToolStatusBadge";

interface ReasoningStreamProps {
  text: string;
  live: boolean;
}

function ReasoningStream({ text, live }: ReasoningStreamProps) {
  const viewportRef = useRef<HTMLDivElement>(null);

  // Auto-scroll pinned to the newest line while thinking streams.
  useEffect(() => {
    const el = viewportRef.current;
    if (live && el) el.scrollTop = el.scrollHeight;
  }, [text, live]);

  return (
    <div
      ref={viewportRef}
      className={cn(
        "text-[13px] leading-5 text-zinc-400",
        live
          ? "max-h-24 overflow-hidden [mask-image:linear-gradient(to_bottom,transparent_0,black_16px,black_100%)]"
          : "max-h-64 overflow-y-auto scrollbar-none",
      )}
    >
      <p className="whitespace-pre-wrap">{text}</p>
    </div>
  );
}

interface Props {
  row: ChainRow;
  isLast: boolean;
}

const SUB_SESSION_TOOLS = new Set([
  "run_sub_session",
  "delegate_to_expert",
  "handoff_to_expert",
  "get_sub_session_result",
]);

function isLiveSubSessionRow(row: ChainRow): boolean {
  if (!row.tool || !SUB_SESSION_TOOLS.has(row.tool)) return false;
  // A blocking delegate has no output while the teammate works — that IS
  // the live window, covered by the pending card built from the input.
  // Result polls render no pending card (see ToolResult), so an outputless
  // poll row has nothing to open.
  if (row.output === undefined)
    return row.state === "running" && row.tool !== "get_sub_session_result";
  if (!row.output || typeof row.output !== "object") return false;
  const status = (row.output as { status?: unknown }).status;
  return (
    typeof status === "string" &&
    ["running", "queued"].includes(status.toLowerCase())
  );
}

export function ChainRowView({ row, isLast }: Props) {
  const [open, setOpen] = useState(row.requiresAction === true);
  const isReasoning = row.category === "reasoning";
  const artifactPanelOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);

  // What the row needs from the user is only known once the output lands, so
  // a live row mounts closed and flips here — the user can still collapse it.
  useEffect(() => {
    if (row.requiresAction) setOpen(true);
  }, [row.requiresAction]);
  // Browser steps carry the page screenshots the artifact panel shows —
  // auto-expand them while the panel is open so the steps are visible
  // from the start.
  useEffect(() => {
    if (row.category === "browser" && artifactPanelOpen) setOpen(true);
  }, [row.category, artifactPanelOpen]);
  // A delegated run's card is the live view of the teammate working —
  // surface it without a click while the sub-session is still going.
  const liveSubSession = isLiveSubSessionRow(row) && !row.supersededSubSession;
  useEffect(() => {
    if (liveSubSession) setOpen(true);
  }, [liveSubSession]);
  // A delegate output frozen at "running" got its done-state label
  // ("Teammate handled:") while the teammate is in fact still working —
  // keep the running label + shimmer until the polled session goes idle.
  const output =
    row.output && typeof row.output === "object"
      ? (row.output as Record<string, unknown>)
      : null;
  const isSubTool = !!row.tool && SUB_SESSION_TOOLS.has(row.tool);
  const effectiveStatus = useSubSessionEffectiveStatus(
    isSubTool && typeof output?.sub_session_id === "string"
      ? output.sub_session_id
      : null,
    isSubTool && typeof output?.status === "string" ? output.status : null,
  );
  // "unknown" means the poll died, not that the teammate finished — the row
  // only has a running label and a done label, and claiming a result landed
  // is the worse of the two guesses. It also keeps this row consistent with
  // the card below it, whose poll caps on its own mount clock.
  const stillWorking =
    isSubTool &&
    row.state === "done" &&
    ["running", "queued", "unknown"].includes(
      effectiveStatus?.toLowerCase() ?? "",
    );
  const liveReasoning =
    isReasoning && row.state === "running" && !!row.reasoningText;
  // An expert being hired/raised has no output until it lands — the skeleton
  // card stands in for it, so the row has something to show while running.
  const pendingExpertChange =
    !!row.tool &&
    EXPERT_CHANGE_TOOLS.has(row.tool) &&
    row.output === undefined &&
    row.state === "running";
  const hasContent = isReasoning
    ? !!row.reasoningText
    : !row.supersededSubSession &&
      ((row.output !== undefined && row.output !== "") ||
        liveSubSession ||
        pendingExpertChange);
  useEffect(() => {
    if (pendingExpertChange) setOpen(true);
  }, [pendingExpertChange]);
  const showContent = liveReasoning || (open && hasContent);
  const rowText = (
    <SwapText
      text={
        stillWorking && row.tool
          ? (getCatalogLabel(row.tool, row.input, "running")?.text ?? row.text)
          : row.text
      }
      shimmer={row.state === "running" || stillWorking}
      className={cn(
        "max-w-full text-sm transition-colors duration-300",
        row.state === "error" ? "text-red-500" : "text-zinc-600",
      )}
    />
  );

  return (
    <div className="flex items-stretch gap-2.5">
      <div className="flex w-7 flex-col items-center">
        <div
          className={cn(
            "relative flex size-7 shrink-0 items-center justify-center rounded-full transition-colors duration-300",
            row.state === "error" ? "bg-red-50" : "bg-zinc-100",
          )}
        >
          <ToolStatusBadge state={row.state} label={row.text}>
            {row.state !== "error" && row.providerIconSrc ? (
              <ProviderIcon src={row.providerIconSrc} row={row} />
            ) : (
              <RowIcon row={row} />
            )}
          </ToolStatusBadge>
        </div>
        {!isLast && (
          <div className="w-px flex-1 origin-top animate-grow-line bg-zinc-200 motion-reduce:animate-none" />
        )}
      </div>
      <div className={cn("min-w-0 flex-1", isLast ? "pb-0" : "pb-3")}>
        {hasContent ? (
          <button
            type="button"
            onClick={() => setOpen(!open)}
            aria-expanded={showContent}
            className="group/row flex h-7 items-center gap-1.5"
          >
            {rowText}
            {!liveReasoning && (
              <Icon
                icon={ArrowDown01Icon}
                size={10}
                className={cn(
                  "shrink-0 text-zinc-300 transition-transform duration-300 ease-out-quint group-hover/row:text-zinc-500",
                  open && "rotate-180",
                )}
              />
            )}
          </button>
        ) : (
          <div className="flex h-7 items-center gap-1.5">{rowText}</div>
        )}
        {row.detail && (
          <p className="animate-fade-in truncate text-xs text-red-400 motion-reduce:animate-none">
            {row.detail}
          </p>
        )}
        <div className={ACCORDION_PANEL + " " + accordionState(showContent)}>
          <div
            aria-hidden={!showContent}
            inert={!showContent ? ("" as unknown as boolean) : undefined}
            className="min-h-0 overflow-hidden"
          >
            <div
              className={cn("px-px pb-px pt-1.5", showContent && PANEL_REVEAL)}
            >
              {isReasoning ? (
                <ReasoningStream
                  text={row.reasoningText ?? ""}
                  live={liveReasoning}
                />
              ) : (
                <ToolResult row={row} />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
