"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { HeldCall } from "@/app/(platform)/home/components/NeedsYou/components/HeldCall";
import { HeldReviewDialog } from "@/app/(platform)/home/components/NeedsYou/components/HeldReviewDialog";
import { isHeldCall } from "@/app/(platform)/home/components/NeedsYou/helpers";
import { ExpertAttentionCard } from "./ExpertAttentionCard";
import { useExpertNeedsYou } from "./useExpertNeedsYou";

interface Props {
  expert: Expert;
  enabled: boolean;
}

/** One card per item, styled like the stack sections in the chat sidebar.
 *  A held AutoPilot call is the same row and detail Home shows. */
export function ExpertNeedsYouSection({ expert, enabled }: Props) {
  const { rows, pendingCount, pendingIDs, decide, held, carousel } =
    useExpertNeedsYou({
      expertId: expert.id,
      enabled,
    });

  if (rows.length === 0) return null;

  return (
    <section aria-label="Needs you" className="flex min-w-0 flex-col">
      {/* No visible heading: the cards say what needs doing. The count is
          still announced for screen readers. */}
      <span role="status" className="sr-only">
        {`${pendingCount} ${pendingCount === 1 ? "item needs" : "items need"} your attention`}
      </span>
      <div className="flex flex-col gap-2">
        {rows.map(({ item, receipt }) =>
          isHeldCall(item) ? (
            <div
              key={item.id}
              className="overflow-hidden rounded-2xl bg-white smooth-shadow-ring-sm"
            >
              <HeldCall
                item={item}
                receipt={receipt}
                status={held.statusOf(item.id)}
                failed={held.hasFailed(item.id)}
                avatarSize={32}
                onOpen={() => carousel.openAt(item.id)}
                onDecide={(approved) => held.decide([item], approved)}
              />
            </div>
          ) : (
            <ExpertAttentionCard
              key={item.id}
              item={item}
              isProcessing={pendingIDs.has(item.id)}
              onDecision={decide}
            />
          ),
        )}
      </div>
      <span className="sr-only" aria-live="polite">
        {held.announcement}
      </span>
      <HeldReviewDialog carousel={carousel} held={held} />
    </section>
  );
}
