"use client";

import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import {
  AnimatePresence,
  domAnimation,
  LazyMotion,
  m,
  useReducedMotion,
} from "framer-motion";
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
  const hasRequiredAction = rows.some((row) => row.requiresAction);
  const open = expanded || isStreaming || hasRequiredAction;
  const windowMode = isStreaming && !expanded;
  // Rows stay mounted while closed so the 0fr collapse can animate.
  const visible = windowMode ? rows.slice(-COLLAPSED_WINDOW) : rows;

  return (
    <LazyMotion features={domAnimation} strict>
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
            inert={open ? undefined : ("" as unknown as boolean)}
            className="min-h-0 overflow-hidden"
          >
            <div className="flex flex-col pl-0.5 pt-2.5">
              <AnimatePresence mode="popLayout">
                {visible.map((row, i) => (
                  <m.div
                    key={row.key}
                    layout={!reducedMotion}
                    initial={
                      reducedMotion ? false : { opacity: 0, y: 8, scale: 0.985 }
                    }
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={
                      reducedMotion
                        ? undefined
                        : { opacity: 0, y: -6, scale: 0.985 }
                    }
                    transition={{
                      opacity: {
                        duration: reducedMotion ? 0 : 0.18,
                        delay: reducedMotion ? 0 : Math.min(i, 6) * 0.035,
                        ease: [0.22, 1, 0.36, 1],
                      },
                      y: {
                        duration: reducedMotion ? 0 : 0.22,
                        delay: reducedMotion ? 0 : Math.min(i, 6) * 0.035,
                        ease: [0.22, 1, 0.36, 1],
                      },
                      scale: {
                        duration: reducedMotion ? 0 : 0.22,
                        delay: reducedMotion ? 0 : Math.min(i, 6) * 0.035,
                        ease: [0.22, 1, 0.36, 1],
                      },
                      layout: {
                        duration: reducedMotion ? 0 : 0.22,
                        ease: [0.22, 1, 0.36, 1],
                      },
                    }}
                  >
                    <div
                      className={open && !windowMode ? PANEL_REVEAL : undefined}
                      style={{ animationDelay: rowStaggerDelay(i) }}
                    >
                      <ChainRowView
                        row={row}
                        isLast={i === visible.length - 1}
                      />
                    </div>
                  </m.div>
                ))}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </LazyMotion>
  );
}
