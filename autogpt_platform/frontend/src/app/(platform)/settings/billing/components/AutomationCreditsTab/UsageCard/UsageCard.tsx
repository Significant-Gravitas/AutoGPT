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

import { EASE_OUT } from "../../../helpers";
import { useUsageCard, type DailyUsageRow } from "./useUsageCard";

interface Props {
  index?: number;
}

export function UsageCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const { usage, isLoading } = useUsageCard();

  if (isLoading) {
    return <Skeleton className="h-[260px] rounded-[18px]" />;
  }

  const max = Math.max(...usage.map((d) => d.amount), 0.01);
  const totalSpent = usage.reduce((sum, d) => sum + d.amount, 0);
  const totalRuns = usage.reduce((sum, d) => sum + d.runs, 0);

  const first = usage[0];
  const middle = usage[Math.floor(usage.length / 2)];
  const last = usage[usage.length - 1];

  return (
    <motion.section
      initial={reduceMotion ? false : { opacity: 0, y: 12 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={
        reduceMotion
          ? undefined
          : { duration: 0.32, ease: EASE_OUT, delay: 0.04 + index * 0.05 }
      }
      className="flex w-full flex-col gap-2"
    >
      <div className="px-4">
        <Text variant="body-medium" as="span" className="text-textBlack">
          Usage
        </Text>
      </div>

      <div className="flex flex-col gap-5 rounded-[18px] border border-zinc-200 bg-white p-5 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div className="flex items-baseline gap-2">
            <Text
              variant="large-medium"
              as="span"
              className="tabular-nums text-textBlack"
            >
              ${totalSpent.toFixed(2)}
            </Text>
            <Text variant="body" as="span" className="text-zinc-500">
              · {totalRuns} runs
            </Text>
          </div>
          <Text variant="body" as="span" className="text-zinc-700">
            Last 30 days
          </Text>
        </div>

        <TooltipProvider delayDuration={150}>
          <div className="flex h-44 items-end gap-1">
            {usage.map((day, i) => (
              <UsageBar
                key={day.date}
                day={day}
                index={i}
                max={max}
                reduceMotion={Boolean(reduceMotion)}
              />
            ))}
          </div>
        </TooltipProvider>

        <div className="flex justify-between">
          <Text variant="small" as="span" className="text-zinc-500">
            {first.date}
          </Text>
          <Text variant="small" as="span" className="text-zinc-500">
            {middle.date}
          </Text>
          <Text variant="small" as="span" className="text-zinc-500">
            {last.date}
          </Text>
        </div>
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

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={`${day.date}: $${day.amount.toFixed(2)}, ${day.runs} runs`}
          className="group flex h-full flex-1 flex-col justify-end focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
        >
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
            className="w-full cursor-help rounded-[3px] bg-zinc-200 transition-colors group-hover:bg-zinc-300"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top">
        <div className="flex flex-col gap-0.5">
          <span className="text-[0.6785rem] font-medium uppercase tracking-[0.06em] text-zinc-500">
            {day.date}
          </span>
          <span className="text-sm font-medium tabular-nums text-textBlack">
            ${day.amount.toFixed(2)}
          </span>
          <span className="text-xs tabular-nums text-zinc-500">
            {day.runs} runs
          </span>
        </div>
      </TooltipContent>
    </Tooltip>
  );
}
