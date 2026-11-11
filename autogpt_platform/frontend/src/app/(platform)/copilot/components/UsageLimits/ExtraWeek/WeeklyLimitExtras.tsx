"use client";

import type { ExtraWeekOffer as Offer } from "@/app/api/__generated__/models/extraWeekOffer";
import { Text } from "@/components/atoms/Text/Text";
import { useReadEveryRender } from "@/hooks/useReadEveryRender";
import { formatExactReset } from "@/services/usageExperience/resetTime";
import { ExtraWeekOffer } from "./ExtraWeekOffer";

interface Props {
  /** The limit that stopped the chat. */
  window: "daily" | "weekly";
  usage?: {
    reserve?: { resets_at: string | Date } | null;
    extra_week_offer?: Offer | null;
  } | null;
  onBought?: () => void;
}

/** What a customer out of their weekly limit can still count on: the small
 *  daily reserve that refills first, and the extra week they can add now.
 *  Nothing for a daily limit: an extra week is used once the week runs out,
 *  so it wouldn't lift today's limit. */
export function WeeklyLimitExtras({ window, usage, onBought }: Props) {
  const refills = useReadEveryRender(() =>
    formatExactReset(usage?.reserve?.resets_at),
  );
  const offer = usage?.extra_week_offer;
  if (window !== "weekly" || (!refills && !offer)) return null;
  return (
    <div className="flex flex-col gap-3">
      {refills && (
        <Text tone="muted" variant="small-loose">
          A small daily reserve refills {refills}.
        </Text>
      )}
      {offer && (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Text variant="small-medium">
            Out of usage? Add another week now.
          </Text>
          <ExtraWeekOffer offer={offer} onBought={onBought} />
        </div>
      )}
    </div>
  );
}
