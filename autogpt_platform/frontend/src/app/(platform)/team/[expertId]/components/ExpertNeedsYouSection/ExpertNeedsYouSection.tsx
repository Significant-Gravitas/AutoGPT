"use client";

import { ExpertAttentionCard } from "./ExpertAttentionCard";
import { useExpertNeedsYou } from "./useExpertNeedsYou";

interface Props {
  expertId: string;
  enabled: boolean;
}

/** One card per item, styled like the stack sections in the chat sidebar. */
export function ExpertNeedsYouSection({ expertId, enabled }: Props) {
  const { items, pendingIDs, decide } = useExpertNeedsYou({
    expertId,
    enabled,
  });

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
