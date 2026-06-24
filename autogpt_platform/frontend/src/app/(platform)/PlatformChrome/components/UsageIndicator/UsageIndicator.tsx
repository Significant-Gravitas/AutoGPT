"use client";

import { GaugeIcon } from "@phosphor-icons/react";
import Link from "next/link";
import { useUsageIndicator } from "./useUsageIndicator";

export function UsageIndicator() {
  const { percent } = useUsageIndicator();
  const label =
    percent !== null ? `Today's usage: ${percent}%` : "Today's usage";

  return (
    <Link
      href="/settings/billing"
      aria-label={label}
      title={label}
      className="relative flex size-8 items-center justify-center rounded-xl border border-zinc-200 bg-zinc-100 p-0 transition-colors hover:bg-zinc-200"
    >
      <GaugeIcon className="size-5 text-black" />

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
            rx="11"
            pathLength={100}
            strokeDasharray={`${percent} 100`}
            strokeLinecap="round"
            strokeWidth={2}
            className="stroke-purple-600"
          />
        </svg>
      ) : null}
    </Link>
  );
}
