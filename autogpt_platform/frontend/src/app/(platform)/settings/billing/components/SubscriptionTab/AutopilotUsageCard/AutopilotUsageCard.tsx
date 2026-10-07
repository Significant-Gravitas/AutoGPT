"use client";

import { motion, useReducedMotion } from "framer-motion";
import { Clock01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { formatUsagePercent } from "@/services/usageExperience/helpers";
import { getSectionMotionProps } from "../../../helpers";
import {
  useAutopilotUsageCard,
  type UsageWindowView,
} from "./useAutopilotUsageCard";

interface Props {
  index?: number;
}

export function AutopilotUsageCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const state = useAutopilotUsageCard();
  if (state.isLoading) return <Skeleton className="h-80 rounded-2xl" />;
  if (
    !state.inactive &&
    (state.isError || (!state.hasUsage && !state.unlimited))
  )
    return (
      <ErrorCard
        context="usage"
        hint="Your usage couldn’t be loaded. Try again in a moment."
        onRetry={state.retry}
        className="h-full"
      />
    );
  return (
    <motion.section
      {...getSectionMotionProps(index, Boolean(reduceMotion))}
      aria-label="Usage"
      className="flex h-full min-h-80 flex-col rounded-2xl border border-zinc-200 bg-white p-6"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Text variant="large-medium" as="h2">
          Usage
        </Text>
        {!state.inactive && (
          <Text
            variant="small"
            className={state.blocked ? "text-orange-700" : "text-zinc-500"}
          >
            {state.blocked ? "Allowance reached" : "On track"}
          </Text>
        )}
      </div>
      {state.inactive || state.unlimited ? (
        <div className="my-auto py-8">
          <Text variant="large-medium" className="mb-2">
            {state.paymentFailed
              ? "Your payment needs attention."
              : state.unlimited
                ? "Room to keep going."
                : "Ready for a fresh start?"}
          </Text>
          <Text variant="small" tone="secondary" className="leading-6">
            {state.paymentFailed
              ? "Review your payment details to continue. Your conversations and agents stay saved."
              : state.unlimited
                ? "Your plan has no daily or weekly usage limits."
                : "Choose a plan to continue working with your saved conversations and agents."}
          </Text>
        </div>
      ) : (
        <>
          {state.featured && <UsageBar window={state.featured} primary />}
          {state.secondary && (
            <div className="mt-6 border-t border-zinc-100 pt-5">
              <UsageBar window={state.secondary} />
            </div>
          )}
          {state.lifetime && (
            <div className="mt-6 border-t border-zinc-100 pt-5">
              <Text variant="small-medium" className="mb-1">
                One allowance for your whole trial
              </Text>
              <Text variant="small" tone="secondary" className="leading-6">
                This allowance lasts for your whole trial. It doesn’t refresh
                each day or week.
              </Text>
            </div>
          )}
        </>
      )}
      <Text
        variant="small"
        tone="secondary"
        className="mt-auto pt-6 text-xs leading-5"
      >
        {state.inactive
          ? "Your work stays saved."
          : "Usage reflects the work your requests need. No surprise overages."}
      </Text>
    </motion.section>
  );
}

function UsageBar({
  window,
  primary = false,
}: {
  window: UsageWindowView;
  primary?: boolean;
}) {
  const percent = Math.min(Math.max(window.percent, 0), 100);
  const label = formatUsagePercent(percent).replace("%", "");
  return (
    <div
      className={primary ? "mt-5" : ""}
      data-testid={primary ? "featured-usage" : undefined}
    >
      <div className="mb-3 flex items-end justify-between gap-3">
        <Text variant="body-medium" className={primary ? "pb-1.5" : ""}>
          {window.label}
        </Text>
        {primary ? (
          <span className="flex items-baseline gap-1 whitespace-nowrap">
            <Text
              variant="h2"
              as="span"
              className="font-sans text-[44px] font-semibold leading-[50px] tracking-[-0.055em]"
            >
              {label}
            </Text>
            <Text as="span" variant="large" tone="secondary">
              %
            </Text>
            <Text as="span" variant="small" tone="secondary" className="ml-1">
              used
            </Text>
            <span className="sr-only">{label}% used</span>
          </span>
        ) : (
          <Text variant="small" tone="secondary">
            {label}% used
          </Text>
        )}
      </div>
      <div
        role="progressbar"
        aria-label={`${window.label} usage`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        className="h-2 overflow-hidden rounded-full bg-zinc-100"
      >
        <div
          className={`h-full rounded-full transition-[width] duration-300 motion-reduce:transition-none ${percent >= 100 ? "bg-orange-500" : "bg-purple-500"}`}
          style={{ width: `${percent}%` }}
        />
      </div>
      <div className="mt-3 flex items-start gap-1.5">
        <Icon
          icon={Clock01Icon}
          size={14}
          className="mt-0.5 shrink-0 text-zinc-500"
        />
        <Text variant="small" tone="secondary" className="text-xs leading-5">
          {window.prefix} {window.value}
        </Text>
      </div>
    </div>
  );
}
