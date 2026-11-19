"use client";

import type { UsageResetOffer as Offer } from "@/app/api/__generated__/models/usageResetOffer";
import { Text } from "@/components/atoms/Text/Text";
import { useReadEveryRender } from "@/hooks/useReadEveryRender";
import { formatExactReset } from "@/services/usageExperience/resetTime";
import { PaidResetOffer } from "./PaidResetOffer";

interface Props {
  /** The limit that stopped the chat. */
  window: "daily" | "weekly";
  usage?: {
    weekly?: { resets_at?: string | Date } | null;
    reserve?: { resets_at: string | Date } | null;
    reset_offer?: Offer | null;
  } | null;
  onBought?: () => void;
}

/** What a customer stopped by a limit can still count on: after a weekly
 *  limit, the small daily reserve that refills first; after either limit, a
 *  paid reset they can buy now. */
export function LimitReachedExtras({ window, usage, onBought }: Props) {
  const refills = useReadEveryRender(() =>
    window === "weekly" ? formatExactReset(usage?.reserve?.resets_at) : null,
  );
  const offer = usage?.reset_offer;
  if (!refills && !offer) return null;
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
            Need to keep going? Reset your daily and weekly limits now.
          </Text>
          <PaidResetOffer
            offer={offer}
            weeklyResetsAt={usage?.weekly?.resets_at}
            onBought={onBought}
          />
        </div>
      )}
    </div>
  );
}
