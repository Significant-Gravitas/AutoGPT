"use client";

import type { ExtraWeekOffer as Offer } from "@/app/api/__generated__/models/extraWeekOffer";
import { ExtraWeekOffer } from "@/app/(platform)/copilot/components/UsageLimits/ExtraWeek/ExtraWeekOffer";
import { Text } from "@/components/atoms/Text/Text";
import { formatUsagePercent } from "@/services/usageExperience/helpers";

interface Props {
  reserve: { percent: number; refills: string | null } | null;
  extra: { percent: number; count: number; expires: string | null } | null;
  offer: Offer | null;
}

/** What limits on the customer's own billing clock add to the usage card:
 *  the daily reserve, any paid extra weeks, and the offer to add one. */
export function UsageExtras({ reserve, extra, offer }: Props) {
  if (!reserve && !extra && !offer) return null;
  return (
    <div className="mt-6 flex flex-col gap-4 border-t border-separator pt-5">
      {reserve && (
        <UsageLine
          label="Daily reserve"
          figure={`${formatUsagePercent(reserve.percent)} used today`}
          note={`A small part of your weekly limit kept back each day, so scheduled work keeps running after the rest is used.${reserve.refills ? ` Refills ${reserve.refills}.` : ""}`}
        />
      )}
      {extra && (
        <UsageLine
          label={extra.count > 1 ? `${extra.count} extra weeks` : "Extra week"}
          figure={`${formatUsagePercent(extra.percent)} used`}
          note={`Used once your weekly limit runs out.${extra.expires ? ` Expires ${extra.expires}.` : ""}`}
        />
      )}
      {offer && (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Text variant="small-loose" tone="secondary">
            Out of usage? Add another week now.
          </Text>
          <ExtraWeekOffer offer={offer} />
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
