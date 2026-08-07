"use client";

import { UsagePopover } from "@/app/(platform)/copilot/components/UsageLimits/UsagePopover/UsagePopover";
import { useUsageIndicator } from "./useUsageIndicator";
import { GaugeIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function UsageIndicator() {
  const { percent } = useUsageIndicator();
  const label =
    percent !== null ? `Today's usage: ${percent}%` : "Today's usage";

  return (
    <UsagePopover
      align="end"
      trigger={
        <button
          type="button"
          aria-label={label}
          className="relative flex size-8 items-center justify-center rounded-lg p-0 transition-colors hover:bg-zinc-100"
        >
          <Icon icon={GaugeIcon} className="size-5 text-black" />

          {percent ? (
            <svg
              viewBox="0 0 32 32"
              fill="none"
              aria-hidden
              className="pointer-events-none absolute inset-0 size-full"
            >
              <rect
                x="1"
                y="1"
                width="30"
                height="30"
                rx="8"
                pathLength={100}
                strokeDasharray={`${percent} 100`}
                strokeLinecap="round"
                strokeWidth={2}
                className="stroke-purple-600"
              />
            </svg>
          ) : null}
        </button>
      }
    />
  );
}
