"use client";

import { motion, useReducedMotion } from "framer-motion";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";

import { EASE_OUT, getSectionMotionProps } from "../../../helpers";
import { useUsageCard, type DailyUsageRow } from "./useUsageCard";

interface Props {
  index?: number;
}

const Y_TICK_COUNT = 4;

export function UsageCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const { usage, isLoading } = useUsageCard();

  const sectionMotion = getSectionMotionProps(index, Boolean(reduceMotion));

  if (isLoading) {
    return (
      <motion.div {...sectionMotion}>
        <Skeleton className="h-[260px] rounded-[18px]" />
      </motion.div>
    );
  }

  const displayMax = Math.max(...usage.map((d) => d.amount), 0);
  const normalizedMax = Math.max(displayMax, 0.01);
  const totalSpent = usage.reduce((sum, d) => sum + d.amount, 0);
  const totalRuns = usage.reduce((sum, d) => sum + d.runs, 0);
  const yTicks = Array.from({ length: Y_TICK_COUNT + 1 }, (_, i) => {
    const value = (displayMax / Y_TICK_COUNT) * (Y_TICK_COUNT - i);
    return { value, label: `$${value.toFixed(2)}` };
  });

  const firstDate = usage.at(0)?.date ?? "—";
  const middleDate = usage[Math.floor(usage.length / 2)]?.date ?? "—";
  const lastDate = usage.at(-1)?.date ?? "—";

  return (
    <motion.section {...sectionMotion} className="flex w-full flex-col gap-2">
      <div className="px-4">
        <Text variant="body-medium" as="span" className="text-textBlack">
          Usage
        </Text>
      </div>

      <div className="flex flex-col gap-5 rounded-[18px] border border-border-base bg-surface p-5 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div className="flex items-baseline gap-2">
            <Text
              variant="large-medium"
              as="span"
              className="tabular-nums text-textBlack"
            >
              ${totalSpent.toFixed(2)}
            </Text>
            <Text variant="body" as="span" className="text-foreground-tertiary">
              · {totalRuns} runs
            </Text>
          </div>
          <Text variant="body" as="span" className="text-foreground-body">
            Last 30 days
          </Text>
        </div>

        <TooltipProvider delayDuration={150}>
          <div className="flex gap-3">
            <div
              className="flex h-44 w-10 flex-col justify-between text-right"
              aria-hidden="true"
            >
              {yTicks.map((tick, i) => (
                <Text
                  key={i}
                  variant="small"
                  as="span"
                  className="tabular-nums text-foreground-subtle"
                >
                  {tick.label}
                </Text>
              ))}
            </div>

            <div className="flex flex-1 flex-col gap-2">
              <div className="relative">
                <div className="relative flex h-44 items-end gap-1 border-b border-l border-border-strong pl-1">
                  {usage.map((day, i) => (
                    <UsageBar
                      key={day.date}
                      day={day}
                      index={i}
                      max={normalizedMax}
                      reduceMotion={Boolean(reduceMotion)}
                    />
                  ))}
                </div>
              </div>

              <div className="flex justify-between pl-1">
                <Text
                  variant="small"
                  as="span"
                  className="text-foreground-tertiary"
                >
                  {firstDate}
                </Text>
                <Text
                  variant="small"
                  as="span"
                  className="text-foreground-tertiary"
                >
                  {middleDate}
                </Text>
                <Text
                  variant="small"
                  as="span"
                  className="text-foreground-tertiary"
                >
                  {lastDate}
                </Text>
              </div>
            </div>
          </div>
        </TooltipProvider>
      </div>
    </motion.section>
  );
}

function UsageBar({
  day,
  index,
  max,
  reduceMotion,
}: {
  day: DailyUsageRow;
  index: number;
  max: number;
  reduceMotion: boolean;
}) {
  const heightPercent = (day.amount / max) * 100;
  const isEmpty = day.amount <= 0;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={
            isEmpty
              ? `${day.date}: $0.00, 0 runs`
              : `${day.date}: $${day.amount.toFixed(2)}, ${day.runs} runs`
          }
          // TODO(tokens): no fitting role for focus-visible:ring-zinc-400 #ADADB3; #ADADB3 is foreground-subtle, a text colour
          className="group flex h-full flex-1 flex-col justify-end focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
        >
          {isEmpty ? (
            <div className="h-px w-full bg-transparent" />
          ) : (
            <motion.div
              initial={
                reduceMotion ? { height: `${heightPercent}%` } : { height: 0 }
              }
              animate={{ height: `${heightPercent}%` }}
              transition={
                reduceMotion
                  ? undefined
                  : {
                      duration: 0.5,
                      ease: EASE_OUT,
                      delay: 0.15 + index * 0.015,
                    }
              }
              className="w-full cursor-help rounded-t-[3px] border-t-2 border-brand bg-brand/30 transition-colors group-hover:bg-brand/50"
            />
          )}
        </button>
      </TooltipTrigger>
      <TooltipContent side="top">
        <div className="flex flex-col gap-0.5">
          <span
            // TODO(tokens): SECRT-3152 rare size (text-[0.6785rem])
            className="text-[0.6785rem] font-medium uppercase tracking-eyebrow text-foreground-subtle"
          >
            {day.date}
          </span>
          <span
            // TODO(tokens): no fitting role for text-zinc-50 #F9F9FA; #F9F9FA is surface-muted, a fill
            className="text-sm font-medium tabular-nums text-zinc-50"
          >
            ${day.amount.toFixed(2)}
          </span>
          <span className="text-xs tabular-nums text-foreground-subtle">
            {day.runs} runs
          </span>
        </div>
      </TooltipContent>
    </Tooltip>
  );
}
