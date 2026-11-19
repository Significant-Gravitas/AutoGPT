"use client";

import { Tick02Icon } from "@hugeicons/core-free-icons";
import { AnimatePresence, m } from "framer-motion";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ChainActionsContext, type ChainActions } from "./chainActions";
import { ChainRowView } from "./ChainRowView";
import type { ChainRow } from "./helpers";

const EASE_OUT_QUINT = [0.22, 1, 0.36, 1] as const;

interface Props {
  rows: ChainRow[];
  isStreaming: boolean;
  /** The streaming window: rows slide in without stagger. */
  windowMode: boolean;
  showDone: boolean;
  reducedMotion: boolean;
  /** Key of the newest row of a live chain, if any. */
  newestLiveKey?: string;
  chainActions: ChainActions;
  readOnly: boolean;
}

/** The chain's rows and its closing "Done" step. Rows present when the chain
 *  first draws come in with its header; only rows that arrive after that
 *  animate on their own. */
export function ChainRows({
  rows,
  isStreaming,
  windowMode,
  showDone,
  reducedMotion,
  newestLiveKey,
  chainActions,
  readOnly,
}: Props) {
  return (
    <>
      <AnimatePresence mode="popLayout" initial={false}>
        {rows.map((row, i) => {
          // While streaming the window slides on every tool, so rows enter
          // without stagger and leave quickly — the stagger is for the
          // settled list revealed on expand. Post-stream mounts (manual
          // expand) skip per-row animation entirely; PANEL_REVEAL fades the
          // panel.
          const stagger =
            reducedMotion || windowMode ? 0 : Math.min(i, 6) * 0.035;
          const animateIn = !reducedMotion && isStreaming;
          const enter = {
            duration: reducedMotion ? 0 : 0.22,
            delay: stagger,
            ease: EASE_OUT_QUINT,
          };
          return (
            <m.div
              key={row.key}
              layout={!reducedMotion}
              initial={animateIn ? { opacity: 0, y: 8, scale: 0.985 } : false}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={
                reducedMotion
                  ? undefined
                  : {
                      opacity: 0,
                      y: -6,
                      scale: 0.985,
                      transition: { duration: 0.12, ease: EASE_OUT_QUINT },
                    }
              }
              transition={{
                opacity: { ...enter, duration: reducedMotion ? 0 : 0.18 },
                y: enter,
                scale: enter,
                layout: {
                  duration: reducedMotion ? 0 : 0.22,
                  ease: EASE_OUT_QUINT,
                },
              }}
            >
              <ChainActionsContext.Provider value={chainActions}>
                <ChainRowView
                  row={row}
                  isLast={i === rows.length - 1 && !showDone}
                  readOnly={readOnly}
                  arrivedFinished={
                    row.key === newestLiveKey && row.state === "done"
                  }
                />
              </ChainActionsContext.Provider>
            </m.div>
          );
        })}
      </AnimatePresence>
      {showDone && (
        <div className="flex items-stretch gap-2.5">
          <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-hover">
            <Icon icon={Tick02Icon} size={14} tone="secondary" />
          </div>
          <div className="flex h-7 items-center text-body text-foreground-secondary">
            Done
          </div>
        </div>
      )}
    </>
  );
}
