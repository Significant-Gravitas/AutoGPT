"use client";

import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { useEffect, useRef, useState } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { useCopilotUIStore } from "../../store";
import { ACCORDION_PANEL, accordionState, PANEL_REVEAL } from "./accordion";
import type { ChainRow } from "./helpers";
import { ProviderIcon, RowIcon } from "./RowIcon";
import { SwapText } from "./SwapText";
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

export function ChainRowView({ row, isLast }: Props) {
  const [open, setOpen] = useState(row.requiresAction === true);
  const isReasoning = row.category === "reasoning";
  const artifactPanelOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);

  // Browser steps carry the page screenshots the artifact panel shows —
  // auto-expand them while the panel is open so the steps are visible
  // from the start.
  useEffect(() => {
    if (row.category === "browser" && artifactPanelOpen) setOpen(true);
  }, [row.category, artifactPanelOpen]);
  const liveReasoning =
    isReasoning && row.state === "running" && !!row.reasoningText;
  const hasContent = isReasoning
    ? !!row.reasoningText
    : row.output !== undefined;
  // Action-required cards (credential setup, review, login) must stay on
  // screen until resolved — the row cannot be collapsed.
  const forcedOpen = row.requiresAction === true && hasContent;
  const showContent = liveReasoning || forcedOpen || (open && hasContent);
  const rowText = (
    <SwapText
      text={row.text}
      shimmer={row.state === "running"}
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
        {hasContent && !forcedOpen ? (
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
            inert={showContent ? undefined : ("" as unknown as boolean)}
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
