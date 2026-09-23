"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertAttentionCard } from "./ExpertAttentionCard";
import { useExpertNeedsYou } from "./useExpertNeedsYou";

interface Props {
  expert: Expert;
  enabled: boolean;
}

/** One card per item, styled like the stack sections in the chat sidebar.
 *  Setup items are left to the Team page's Setup needed card, which names
 *  the missing connection and offers the fix. */
export function ExpertNeedsYouSection({ expert, enabled }: Props) {
  const {
    items: allItems,
    pendingIDs,
    decide,
  } = useExpertNeedsYou({
    expertId: expert.id,
    enabled,
  });
  const items = allItems.filter((item) => item.kind !== "setup");

  if (items.length === 0) return null;

  return (
    <section aria-label="Needs you" className="flex min-w-0 flex-col">
      {/* No visible heading: the cards say what needs doing. The count is
          still announced for screen readers. */}
      <span role="status" className="sr-only">
        {`${items.length} ${items.length === 1 ? "item needs" : "items need"} your attention`}
      </span>
      <div className="flex flex-col gap-2">
        {items.map((item) => (
          <ExpertAttentionCard
            key={item.id}
            item={item}
            isProcessing={pendingIDs.has(item.id)}
            onDecision={decide}
          />
        ))}
      </div>
    </section>
  );
}
