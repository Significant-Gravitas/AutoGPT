"use client";

import { motion, useReducedMotion } from "framer-motion";

import { Text } from "@/components/atoms/Text/Text";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";

import { EASE_OUT, getSectionMotionProps } from "../../../helpers";
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
      {...getSectionMotionProps(index, Boolean(reduceMotion))}
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
  const isHigh = percent >= 80;
  const percentLabel =
    window.percent > 0 && percent === 0 ? "<1% used" : `${percent}% used`;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <Text variant="body-medium" className="text-neutral-700">
          {window.label}
        </Text>
        <Text variant="body" className="tabular-nums text-neutral-500">
          {percentLabel}
        </Text>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200">
        <motion.div
          role="progressbar"
          aria-label={`${window.label} usage`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          initial={reduceMotion ? { width: `${percent}%` } : { width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={
            reduceMotion
              ? undefined
              : { duration: 0.9, ease: EASE_OUT, delay: 0.25 }
          }
          className={`h-full rounded-full ${
            isHigh ? "bg-orange-500" : "bg-blue-500"
          }`}
        />
      </div>
      <Text variant="small" className="text-neutral-400">
        {window.prefix} <span className="text-neutral-700">{window.value}</span>
      </Text>
    </div>
  );
}
