"use client";

import { motion, useReducedMotion } from "framer-motion";

import { Text } from "@/components/atoms/Text/Text";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";

import { EASE_OUT } from "../../../helpers";
import {
  useAutopilotUsageCard,
  type UsageWindowView,
} from "./useAutopilotUsageCard";

interface Props {
  index?: number;
}

const USAGE_EXPLAINER =
  "Each Autopilot request consumes a share of your plan's allowance based on the work performed. Simple requests use little; complex workflows use more. No surprise overages.";

export function AutopilotUsageCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const { today, week, hasUsage } = useAutopilotUsageCard();

  if (!hasUsage) return null;

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
      <div className="flex items-center gap-1 px-4">
        <Text variant="body-medium" as="span" className="text-textBlack">
          Autopilot usage
        </Text>
        <InformationTooltip description={USAGE_EXPLAINER} iconSize={22} />
      </div>

      <div className="flex flex-col gap-6 rounded-[18px] border border-zinc-200 bg-white px-5 py-5 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        {today ? <UsageBar window={today} /> : null}
        {week ? <UsageBar window={week} /> : null}
      </div>
    </motion.section>
  );
}

function UsageBar({ window }: { window: UsageWindowView }) {
  const reduceMotion = useReducedMotion();
  const percent = Math.min(Math.max(window.percent, 0), 100);

  return (
    <div className="flex flex-col gap-2">
      <Text variant="body-medium" as="span" className="text-textBlack">
        {window.label}
      </Text>
      <div className="flex items-center gap-3">
        <div className="relative h-6 flex-1 overflow-hidden rounded-[4px] bg-[repeating-linear-gradient(135deg,_#d4d4d8_0,_#d4d4d8_2px,_transparent_2px,_transparent_6px)]">
          <motion.div
            initial={reduceMotion ? { width: `${percent}%` } : { width: 0 }}
            animate={{ width: `${percent}%` }}
            transition={
              reduceMotion
                ? undefined
                : { duration: 0.9, ease: EASE_OUT, delay: 0.25 }
            }
            className="h-full bg-zinc-300"
          />
        </div>
        <Text
          variant="body-medium"
          as="span"
          className="shrink-0 tabular-nums text-textBlack"
        >
          {percent}%
        </Text>
      </div>
      <Text variant="body" as="span" className="text-zinc-500">
        {window.prefix} <span className="text-zinc-800">{window.value}</span>
      </Text>
    </div>
  );
}
