"use client";

import type { UsageResetOffer as Offer } from "@/app/api/__generated__/models/usageResetOffer";
import { PaidResetOffer } from "@/app/(platform)/copilot/components/UsageLimits/PaidReset/PaidResetOffer";
import { Text } from "@/components/atoms/Text/Text";
import { formatUsagePercent } from "@/services/usageExperience/helpers";

interface Props {
  reserve: { percent: number; refills: string | null } | null;
  /** A paid reset in force, and the weekly reset it lasts until. */
  resetActive: { until: string | null } | null;
  /** Sent only while a limit is reached. */
  offer: Offer | null;
  weeklyResetsAt?: string | Date | null;
}

/** What limits on the customer's own billing clock add to the usage card:
 *  the daily reserve, a paid reset in force, and the offer to buy one when
 *  a limit is reached. */
export function UsageExtras({
  reserve,
  resetActive,
  offer,
  weeklyResetsAt,
}: Props) {
  if (!reserve && !resetActive && !offer) return null;
  return (
    <div className="mt-6 flex flex-col gap-4 border-t border-separator pt-5">
      {reserve && (
        <UsageLine
          label="Daily reserve"
          figure={`${formatUsagePercent(reserve.percent)} used today`}
          note={`A small part of your weekly limit kept back each day, so scheduled work keeps running after the rest is used.${reserve.refills ? ` Refills ${reserve.refills}.` : ""}`}
        />
      )}
      {resetActive && (
        <UsageLine
          label="Reset active"
          figure="No daily limit"
          note={`Your daily limit is lifted until your weekly reset${resetActive.until ? ` ${resetActive.until}` : ""}. The usage above is what you’ve used since your reset.`}
        />
      )}
      {offer && (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Text variant="small-loose" tone="secondary">
            Need to keep going? Reset your daily and weekly limits now.
          </Text>
          <PaidResetOffer offer={offer} weeklyResetsAt={weeklyResetsAt} />
        </div>
      )}
    </div>
  );
}

function UsageLine({
  label,
  figure,
  note,
}: {
  label: string;
  figure: string;
  note: string;
}) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-3">
        <Text variant="small-medium">{label}</Text>
        <Text variant="small" tone="secondary">
          {figure}
        </Text>
      </div>
      <Text variant="small-loose" tone="secondary">
        {note}
      </Text>
    </div>
  );
}
