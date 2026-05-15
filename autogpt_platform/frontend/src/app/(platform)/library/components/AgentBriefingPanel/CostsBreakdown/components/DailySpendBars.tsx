"use client";

import type { UserDailyCost } from "@/app/api/__generated__/models/userDailyCost";
import { Text } from "@/components/atoms/Text/Text";
import { formatCents } from "@/app/(platform)/copilot/components/usageHelpers";
import { formatShortDate } from "../helpers";

interface Props {
  daily: UserDailyCost[];
}

export function DailySpendBars({ daily }: Props) {
  if (daily.length === 0) return null;

  const maxCost = Math.max(...daily.map((d) => d.cost_cents), 1);

  return (
    <section className="flex flex-col gap-2">
      <Text variant="body-medium" className="text-neutral-800">
        Daily spend
      </Text>
      <div
        className="flex h-24 items-end gap-1"
        role="list"
        aria-label="Daily spend bars"
      >
        {daily.map((bucket) => {
          const heightPct = Math.round((bucket.cost_cents / maxCost) * 100);
          const visibleHeight = Math.max(
            bucket.cost_cents > 0 ? 4 : 1,
            heightPct,
          );
          const key =
            bucket.date instanceof Date
              ? bucket.date.toISOString()
              : String(bucket.date);
          return (
            <div
              key={key}
              role="listitem"
              className="group relative flex flex-1 flex-col items-center justify-end"
              title={`${formatShortDate(bucket.date)} · ${formatCents(bucket.cost_cents)} · ${bucket.run_count} run${bucket.run_count === 1 ? "" : "s"}`}
            >
              <div
                className="w-full rounded-t-sm bg-blue-500 transition-opacity group-hover:opacity-80"
                style={{ height: `${visibleHeight}%` }}
              />
            </div>
          );
        })}
      </div>
      <div className="flex justify-between">
        <Text variant="small" className="text-neutral-400">
          {formatShortDate(daily[0].date)}
        </Text>
        <Text variant="small" className="text-neutral-400">
          {formatShortDate(daily[daily.length - 1].date)}
        </Text>
      </div>
    </section>
  );
}
