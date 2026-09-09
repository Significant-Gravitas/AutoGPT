"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { AlertCircleIcon } from "@hugeicons/core-free-icons";
import { ExpertAttentionCard } from "./ExpertAttentionCard";
import { useExpertNeedsYou } from "./useExpertNeedsYou";

interface Props {
  expertId: string;
  enabled: boolean;
}

/** Label outside, one card per item: the same shape as the stack sections in
 *  the chat sidebar. */
export function ExpertNeedsYouSection({ expertId, enabled }: Props) {
  const { items, pendingIDs, decide } = useExpertNeedsYou({
    expertId,
    enabled,
  });

  if (items.length === 0) return null;

  return (
    <section aria-label="Needs you" className="flex min-w-0 flex-col">
      <div className="mb-1.5 flex items-center gap-1.5 px-3.5">
        <Icon
          icon={AlertCircleIcon}
          size={14}
          className="text-zinc-500"
          aria-hidden
        />
        <Text variant="small-medium" as="h2" className="!text-zinc-700">
          Needs you{" "}
          <span
            role="status"
            aria-label={`${items.length} ${items.length === 1 ? "item needs" : "items need"} your attention`}
            className="tabular-nums"
          >
            ({items.length})
          </span>
        </Text>
      </div>
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
