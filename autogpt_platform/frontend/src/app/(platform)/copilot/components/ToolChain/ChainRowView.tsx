"use client";

import { CaretDownIcon } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";
import { ACCORDION_PANEL, accordionState, PANEL_REVEAL } from "./accordion";
import type { ChainRow } from "./helpers";
import { ProviderIcon, RowIcon } from "./RowIcon";
import { SwapIcon, SwapText } from "./SwapText";
import { ToolResult } from "./ToolResult";

function ReasoningStream({ text, live }: { text: string; live: boolean }) {
  const viewportRef = useRef<HTMLDivElement>(null);

  // Auto-scroll pinned to the newest line while thinking streams.
  useEffect(() => {
    const el = viewportRef.current;
    if (live && el) el.scrollTop = el.scrollHeight;
  }, [text, live]);

  return (
    <div
      ref={viewportRef}
      className={
        "text-[13px] leading-5 text-zinc-400 " +
        (live
          ? "max-h-24 overflow-hidden [mask-image:linear-gradient(to_bottom,transparent_0,black_16px,black_100%)]"
          : "max-h-64 overflow-y-auto scrollbar-none")
      }
    >
      <p className="whitespace-pre-wrap">{text}</p>
    </div>
  );
}

export function ChainRowView({
  row,
  isLast,
}: {
  row: ChainRow;
  isLast: boolean;
}) {
  // Question rows start open — they need the user's answer to proceed.
  const [open, setOpen] = useState(row.category === "question");
  const isReasoning = row.category === "reasoning";
  const liveReasoning =
    isReasoning && row.state === "running" && !!row.reasoningText;
  const hasContent = isReasoning
    ? !!row.reasoningText
    : row.output !== undefined;
  const showContent = liveReasoning || (open && hasContent);

  return (
    <div className="flex items-stretch gap-2.5">
      <div className="flex w-7 flex-col items-center">
        <div
          className={
            "relative flex size-7 shrink-0 items-center justify-center rounded-full transition-colors duration-300 " +
            (row.state === "error"
              ? "bg-red-50"
              : row.state === "running"
                ? "bg-purple-50"
                : "bg-zinc-100")
          }
        >
          {row.state === "running" && (
            <span className="absolute inset-0 animate-[spin_0.6s_linear_infinite] rounded-full border border-purple-200 border-t-purple-600 motion-reduce:animate-none" />
          )}
          <SwapIcon
            swapKey={
              row.state === "error" ? "error" : (row.providerIconSrc ?? "icon")
            }
          >
            {row.state !== "error" && row.providerIconSrc ? (
              <ProviderIcon src={row.providerIconSrc} row={row} />
            ) : (
              <RowIcon row={row} />
            )}
          </SwapIcon>
        </div>
        {!isLast && <div className="w-px flex-1 bg-zinc-200" />}
      </div>
      <div className={"min-w-0 flex-1 " + (isLast ? "pb-0" : "pb-3")}>
        <button
          type="button"
          onClick={hasContent ? () => setOpen(!open) : undefined}
          aria-expanded={showContent}
          className={
            "group/row flex h-7 items-center gap-1.5 " +
            (hasContent ? "" : "cursor-default")
          }
        >
          <SwapText
            text={row.text}
            shimmer={row.state === "running"}
            className={
              "max-w-full text-sm transition-colors duration-300 " +
              (row.state === "error" ? "text-red-500" : "text-zinc-600")
            }
          />
          {hasContent && !liveReasoning && (
            <CaretDownIcon
              size={10}
              weight="bold"
              className={
                "shrink-0 text-zinc-300 transition-transform duration-300 ease-out-quint group-hover/row:text-zinc-500 " +
                (open ? "rotate-180" : "")
              }
            />
          )}
        </button>
        {row.detail && (
          <p className="truncate text-xs text-red-400 duration-200 animate-in fade-in motion-reduce:animate-none">
            {row.detail}
          </p>
        )}
        <div className={ACCORDION_PANEL + " " + accordionState(showContent)}>
          <div className="min-h-0 overflow-hidden">
            <div
              className={
                "px-px pb-px pt-1.5 " + (showContent ? PANEL_REVEAL : "")
              }
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
