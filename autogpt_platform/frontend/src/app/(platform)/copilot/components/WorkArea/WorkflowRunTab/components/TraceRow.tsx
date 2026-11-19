"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import type { CSSProperties } from "react";
import { FOCUS } from "../../../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { formatDuration, type TraceStep } from "../trace";
import styles from "../trace.module.css";
import { StatusMark } from "./StatusMark";
import { StepDetail } from "./StepDetail";

/** The row's columns, shared with the axis above the rows. */
export const TRACE_GRID =
  "grid grid-cols-[1rem_minmax(0,7.25rem)_minmax(0,1fr)_2.75rem] items-center gap-2";

const EASE: [number, number, number, number] = [0.22, 1, 0.36, 1];
const WAITING_WIDTH = "14px";

interface Props {
  step: TraceStep;
  index: number;
  totalMs: number;
  ticks: number[];
  isOpen: boolean;
  onToggle: () => void;
}

function percent(ms: number, totalMs: number) {
  return `${Math.min(100, (ms / totalMs) * 100)}%`;
}

/** One block run: its mark and name, a bar on the shared axis, how long it
 *  took. Clicking it opens what went in and came out, in place. */
export function TraceRow({
  step,
  index,
  totalMs,
  ticks,
  isOpen,
  onToggle,
}: Props) {
  const reduceMotion = useReducedMotion();
  const isWaiting = step.state === "waiting";
  const hasDetail = !isWaiting || step.input !== null;

  return (
    <li
      data-step-state={step.state}
      style={{ "--i": index } as CSSProperties}
      className={cn(
        styles.row,
        styles[step.kind],
        styles[step.state],
        "relative not-first:before:absolute not-first:before:-top-px not-first:before:right-2 not-first:before:left-8 not-first:before:h-px not-first:before:bg-border-hairline",
      )}
    >
      <button
        type="button"
        aria-expanded={hasDetail ? isOpen : undefined}
        disabled={!hasDetail}
        onClick={onToggle}
        className={cn(
          TRACE_GRID,
          "h-9 w-full rounded-lg px-2 text-left transition-colors duration-100 ease-out enabled:hover:bg-surface-tray",
          FOCUS,
        )}
      >
        <StatusMark state={step.state} />
        <span
          className={cn(
            styles.tag,
            "flex h-5 w-fit max-w-full min-w-0 items-center rounded-md px-1.5",
          )}
        >
          <Text
            variant="small-medium"
            as="span"
            tone="inherit"
            className="truncate"
          >
            {step.name}
          </Text>
        </span>
        <span className="relative h-5 w-full">
          {ticks.map((at) => (
            <span
              key={at}
              aria-hidden
              style={{ "--at": percent(at, totalMs) } as CSSProperties}
              className={cn(styles.tick, "absolute inset-y-0 left-(--at) w-px")}
            />
          ))}
          <span
            aria-hidden
            style={
              {
                "--left": percent(step.offsetMs, totalMs),
                "--width": isWaiting
                  ? WAITING_WIDTH
                  : percent(step.durationMs, totalMs),
              } as CSSProperties
            }
            className={cn(
              styles.bar,
              step.state === "running" && styles.running,
              "absolute inset-y-0 my-auto h-3.5",
            )}
          >
            <span
              className={cn(
                styles.fill,
                "absolute inset-0 overflow-hidden rounded-md",
              )}
            />
            {step.state === "running" ? (
              <span
                className={cn(
                  styles.tip,
                  "absolute inset-y-0 -right-1 my-auto size-2 rounded-full",
                )}
              />
            ) : null}
          </span>
        </span>
        <Text
          variant="small"
          as="span"
          tone={isWaiting ? "faint" : "quiet"}
          align="right"
          tabular
        >
          {isWaiting ? "–" : formatDuration(step.durationMs)}
        </Text>
      </button>
      <AnimatePresence initial={false}>
        {isOpen ? (
          <motion.div
            key="detail"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: reduceMotion ? 0 : 0.22, ease: EASE }}
            className="overflow-hidden"
          >
            <StepDetail step={step} />
          </motion.div>
        ) : null}
      </AnimatePresence>
    </li>
  );
}
