"use client";
import type { CSSProperties } from "react";

import { motion, useReducedMotion } from "framer-motion";
import { Clock01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { formatUsagePercent } from "@/services/usageExperience/helpers";
import { getSectionMotionProps } from "../../../helpers";
import {
  useAutopilotUsageCard,
  type UsageWindowView,
} from "./useAutopilotUsageCard";
import { UsageUnavailable } from "./UsageUnavailable";

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
    return <UsageUnavailable onRetry={() => void state.retry()} />;
  return (
    <motion.section
      {...getSectionMotionProps(index, Boolean(reduceMotion))}
      aria-label="Usage"
      className="flex h-full min-h-80 flex-col rounded-2xl border border-border-base bg-surface p-6"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Text variant="large-medium" as="h2">
          Usage
        </Text>
        {!state.inactive && (
          <Text variant="small" tone={state.status.tone}>
            {state.status.label}
          </Text>
        )}
      </div>
      {state.inactive || state.unlimited ? (
        <div className="my-auto py-8">
          <Text variant="large-medium" className="mb-2">
            {state.unlimited
              ? "No usage limit"
              : "Choose a plan to keep working"}
          </Text>
          {/* oxlint-disable-next-line design-system/no-restyle -- caption copy at a 24px line height; no Text variant has it */}
          <Text variant="small" tone="secondary" className="leading-6">
            {state.unlimited
              ? "Your plan has no daily or weekly usage limits."
              : "Choose a plan to continue working with your saved conversations and workflows."}
          </Text>
        </div>
      ) : (
        <>
          {state.featured && <UsageBar window={state.featured} primary />}
          {state.secondary && (
            <div className="mt-6 border-t border-separator pt-5">
              <UsageBar window={state.secondary} />
            </div>
          )}
          {state.lifetime && (
            <div className="mt-6 border-t border-separator pt-5">
              <Text variant="small-medium" className="mb-1">
                One allowance for your whole trial
              </Text>
              {/* oxlint-disable-next-line design-system/no-restyle -- caption copy at a 24px line height; no Text variant has it */}
              <Text variant="small" tone="secondary" className="leading-6">
                This allowance lasts for your whole trial. It doesn’t refresh
                each day or week.
              </Text>
            </div>
          )}
        </>
      )}
      <div className="mt-auto pt-6">
        <Text variant="small-loose" tone="secondary">
          {state.inactive
            ? "Your work stays saved."
            : "Bigger requests use more of your allowance. Hitting the limit never adds a charge."}
        </Text>
      </div>
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
        <Text variant="body-medium" className={primary ? "mb-1.5" : ""}>
          {window.label}
        </Text>
        {primary ? (
          <span className="flex items-baseline gap-1 whitespace-nowrap">
            <Text
              variant="h2"
              as="span"
              // oxlint-disable-next-line design-system/no-restyle -- the usage figure is a sans display number; no Text variant has it
              className="leading-12 font-sans text-display font-semibold tracking-display"
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
        className="h-2 overflow-hidden rounded-full bg-surface-hover"
      >
        <div
          // oxlint-disable-next-line design-system/no-raw-colors -- the full bar stays orange until the orange -> warning hue change is approved
          className={`h-full w-[var(--usage-percent)] rounded-full transition-[width] duration-300 motion-reduce:transition-none ${percent >= 100 ? "bg-orange-500" : "bg-brand"}`}
          style={{ "--usage-percent": `${percent}%` } as CSSProperties}
        />
      </div>
      <div className="mt-3 flex items-start gap-1.5 text-foreground-tertiary">
        <Icon icon={Clock01Icon} size={14} className="mt-0.5 shrink-0" />
        <Text variant="small-loose" tone="secondary">
          {window.prefix} {window.value}
        </Text>
      </div>
    </div>
  );
}
