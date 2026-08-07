"use client";

import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
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
import { ACCORDION_PANEL, accordionState, PANEL_REVEAL } from "./accordion";
import { ChainRowView } from "./ChainRowView";
import { type ChainRow, getChainHeading, toChainRow } from "./helpers";
import { SwapText } from "./SwapText";

const COLLAPSED_WINDOW = 2;

interface Props {
  parts: MessagePart[];
  isStreaming: boolean;
}

export function ToolChain({ parts, isStreaming }: Props) {
  const [manualExpanded, setManualExpanded] = useState<boolean | null>(null);
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

  const expanded = manualExpanded === true;
  const heading = getChainHeading(rows, isStreaming && !expanded);
  const hasError = rows.some((row) => row.state === "error");
  const hasRequiredAction = rows.some((row) => row.requiresAction);
  // Auto-open while streaming or action-required; a manual toggle overrides
  // either direction and sticks until the next toggle.
  const open = manualExpanded ?? (isStreaming || hasRequiredAction);
  const windowMode = isStreaming && !expanded;
  // Rows stay mounted while closed so the 0fr collapse can animate.
  const visible = windowMode ? rows.slice(-COLLAPSED_WINDOW) : rows;

  return (
    <LazyMotion features={domAnimation} strict>
      <div className="my-2">
        <button
          type="button"
          onClick={() => setManualExpanded(!open)}
          aria-expanded={open}
          aria-controls={panelId}
          className="group/chain -mx-2 flex w-fit max-w-full items-center gap-1.5 rounded-lg px-2 py-1 text-left transition-colors duration-100 hover:bg-zinc-100"
        >
          <SwapText
            text={heading}
            shimmer={isStreaming && !expanded}
            className={
              "min-w-0 text-sm font-normal " +
              (hasError && !isStreaming ? "text-red-500" : "text-zinc-700")
            }
          />
          <Icon
            icon={ArrowDown01Icon}
            size={12}
            className={
              "shrink-0 text-zinc-400 transition-transform duration-300 ease-out-quint " +
              (open ? "rotate-180" : "")
            }
          />
        </button>
        <div className={ACCORDION_PANEL + " " + accordionState(open)}>
          <div
            id={panelId}
            aria-hidden={!open}
            inert={open ? undefined : ("" as unknown as boolean)}
            className="min-h-0 overflow-hidden"
          >
            <div
              className={
                "flex flex-col pl-0.5 pt-2.5" +
                (open && !windowMode ? " " + PANEL_REVEAL : "")
              }
            >
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
                    <ChainRowView row={row} isLast={i === visible.length - 1} />
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
