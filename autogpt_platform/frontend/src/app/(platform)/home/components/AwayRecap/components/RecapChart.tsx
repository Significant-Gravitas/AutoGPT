"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipPortal,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { cn } from "@/lib/utils";
import { motion, useReducedMotion } from "framer-motion";
import { useId, type CSSProperties } from "react";
import {
  activityChart,
  CHART_HEIGHT,
  CHART_WIDTH,
  LANES,
  positionOf,
  type RecapItem,
} from "../helpers";

interface Props {
  items: RecapItem[];
  start: number;
  end: number;
  startLabel: string;
  timeOf: (item: RecapItem) => string;
  activeID: string | null;
  onActive: (id: string | null) => void;
  selectedID: string | null;
  onSelect: (id: string) => void;
}

const DRAW_SECONDS = 1.4;
const DRAW_EASE = [0.32, 0.72, 0, 1] as const;

// The stretch since the user left as one line: it rises wherever something
// happened, with a dot on each event. On open it draws from left to right
// and each dot lands as the line reaches it.
export function RecapChart({
  items,
  start,
  end,
  startLabel,
  timeOf,
  activeID,
  onActive,
  selectedID,
  onSelect,
}: Props) {
  const prefersReducedMotion = useReducedMotion();
  const gradientID = useId();
  const dots = items.flatMap((item) => {
    const t = positionOf(item, start, end);
    return t === null ? [] : [{ item, t }];
  });
  const chart = activityChart(dots.map((dot) => dot.t));

  return (
    <div className="px-3 pt-1 pb-2">
      <div className="relative h-24 w-full">
        <motion.div
          className="absolute inset-0 text-foreground-ink"
          initial={
            prefersReducedMotion ? false : { clipPath: "inset(0 100% 0 0)" }
          }
          animate={{ clipPath: "inset(0 0% 0 0)" }}
          transition={{ duration: DRAW_SECONDS, ease: DRAW_EASE }}
        >
          <svg
            viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
            preserveAspectRatio="none"
            className="size-full overflow-visible"
            aria-hidden
          >
            <defs>
              <linearGradient id={gradientID} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="currentColor" stopOpacity="0.1" />
                <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
              </linearGradient>
            </defs>
            <path d={chart.area} fill={`url(#${gradientID})`} />
            <path
              d={chart.line}
              fill="none"
              stroke="currentColor"
              strokeOpacity="0.55"
              strokeWidth="1.5"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          </svg>
        </motion.div>

        {dots.map(({ item, t }) => {
          const lane = LANES.find((entry) => entry.kind === item.kind);
          const isActive = activeID === item.id || selectedID === item.id;
          const dot = (
            <span
              className={cn(
                "block size-2 rounded-full ring-2 ring-surface transition-transform duration-150 ease-out",
                lane?.dot,
                isActive && "scale-[1.75]",
              )}
            />
          );
          return (
            <Tooltip key={item.id}>
              <TooltipTrigger asChild>
                <motion.span
                  className="absolute top-(--dot-y) left-(--dot-x) -translate-x-1/2 -translate-y-1/2 p-1.5"
                  style={
                    {
                      "--dot-x": `${t * 100}%`,
                      "--dot-y": `${chart.heightAt(t) * 100}%`,
                    } as CSSProperties
                  }
                  initial={
                    prefersReducedMotion ? false : { scale: 0, opacity: 0 }
                  }
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{
                    delay: prefersReducedMotion ? 0 : t * DRAW_SECONDS,
                    type: "spring",
                    duration: 0.4,
                    bounce: 0.3,
                  }}
                  onMouseEnter={() => onActive(item.id)}
                  onMouseLeave={() => onActive(null)}
                >
                  <button
                    type="button"
                    aria-label={`${lane?.label}: ${item.title}`}
                    aria-pressed={selectedID === item.id}
                    onClick={() => onSelect(item.id)}
                    className="block cursor-pointer rounded-full outline-none focus-visible:ring-2 focus-visible:ring-focus-inbox"
                  >
                    {dot}
                  </button>
                </motion.span>
              </TooltipTrigger>
              <TooltipPortal>
                <TooltipContent side="top">
                  <span className="flex flex-col gap-0.5">
                    <Text variant="small-medium" as="span" tone="inherit">
                      {item.title}
                    </Text>
                    {item.detail ? (
                      <span className="opacity-80">
                        <Text
                          variant="small"
                          as="span"
                          tone="inherit"
                          className="line-clamp-2"
                        >
                          {item.detail}
                        </Text>
                      </span>
                    ) : null}
                    <span className="opacity-60">
                      <Text variant="small" as="span" tone="inherit">
                        {lane?.label} · {timeOf(item)}
                      </Text>
                    </span>
                  </span>
                </TooltipContent>
              </TooltipPortal>
            </Tooltip>
          );
        })}
      </div>
      <div className="mt-1 flex justify-between text-caption text-foreground-faint">
        <span>{startLabel}</span>
        <span>Now</span>
      </div>
    </div>
  );
}
