"use client";

import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useId, useMemo, useState } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import type { MessagePart } from "../ChatMessagesContainer/helpers";
import {
  ACCORDION_PANEL,
  accordionState,
  PANEL_REVEAL,
  rowStaggerDelay,
} from "./accordion";
import { ChainRowView } from "./ChainRowView";
import { type ChainRow, getChainHeading, toChainRow } from "./helpers";
import { SwapText } from "./SwapText";

const COLLAPSED_WINDOW = 2;

interface Props {
  parts: MessagePart[];
  isStreaming: boolean;
}

export function ToolChain({ parts, isStreaming }: Props) {
  const [expanded, setExpanded] = useState(false);
  // A chain holding a question must not collapse away the thing the user
  // has to answer — it stays pinned open while a question row exists.
  const hasQuestion = parts.some((part) => part.type === "tool-ask_question");
  const panelId = useId();
  const reducedMotion = useReducedMotion();

  const rows = useMemo(
    () =>
      parts
        .map((part, i) => toChainRow(part, i))
        .filter((row): row is ChainRow => row !== null),
    [parts],
  );
  if (rows.length === 0) return null;

  const heading = getChainHeading(rows, isStreaming && !expanded);
  const hasError = rows.some((row) => row.state === "error");
  const open = expanded || isStreaming || hasQuestion;
  // Sliding-window mode (streaming, collapsed) animates row turnover with
  // framer; everywhere else rows render static so the accordion's grid
  // height transition stays the only motion.
  const windowMode = isStreaming && !expanded;
  // Rows stay mounted while closed so the 0fr collapse can animate.
  const visible = windowMode ? rows.slice(-COLLAPSED_WINDOW) : rows;

  return (
    <div className="my-2">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={open}
        aria-controls={panelId}
        className="group/chain flex w-full items-center gap-1.5 text-left"
      >
        <Icon
          icon={ArrowRight01Icon}
          size={12}
          className={
            "shrink-0 text-zinc-400 transition-transform duration-300 ease-out-quint " +
            (open ? "rotate-90" : "")
          }
        />
        <SwapText
          text={heading}
          shimmer={isStreaming && !expanded}
          className={
            "min-w-0 text-sm font-normal " +
            (hasError && !isStreaming ? "text-red-500" : "text-zinc-700")
          }
        />
        <span className="ml-auto shrink-0 text-xs text-zinc-400 opacity-0 transition-opacity group-hover/chain:opacity-100">
          {rows.length} steps
        </span>
      </button>
      <div className={ACCORDION_PANEL + " " + accordionState(open)}>
        <div
          id={panelId}
          aria-hidden={!open}
          inert={!open}
          className="min-h-0 overflow-hidden"
        >
          <div className="flex flex-col pl-0.5 pt-2.5">
            {windowMode ? (
              <AnimatePresence initial={false} mode="popLayout">
                {visible.map((row, i) => (
                  <motion.div
                    key={row.key}
                    layout={!reducedMotion}
                    initial={reducedMotion ? false : { opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={reducedMotion ? undefined : { opacity: 0, y: -10 }}
                    transition={{ duration: 0.22, ease: [0.33, 1, 0.68, 1] }}
                  >
                    <ChainRowView row={row} isLast={i === visible.length - 1} />
                  </motion.div>
                ))}
              </AnimatePresence>
            ) : (
              visible.map((row, i) => (
                // Toggling the class off on collapse is what lets the cascade
                // replay on the next expand.
                <div
                  key={row.key}
                  className={open ? PANEL_REVEAL : undefined}
                  style={{ animationDelay: rowStaggerDelay(i) }}
                >
                  <ChainRowView row={row} isLast={i === visible.length - 1} />
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
