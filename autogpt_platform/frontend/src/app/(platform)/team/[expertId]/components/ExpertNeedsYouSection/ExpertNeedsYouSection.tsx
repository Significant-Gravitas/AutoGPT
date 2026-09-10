"use client";

import { AttentionRow } from "@/app/(platform)/home/components/NeedsYou/components/AttentionRow";
import { HomeTile } from "@/app/(platform)/home/components/HomeTile/HomeTile";
import { Text } from "@/components/atoms/Text/Text";
import { AlertCircleIcon } from "@hugeicons/core-free-icons";
import { useExpertNeedsYou } from "./useExpertNeedsYou";

interface Props {
  expertId: string;
  enabled: boolean;
}

export function ExpertNeedsYouSection({ expertId, enabled }: Props) {
  const { items, pendingIDs, decide } = useExpertNeedsYou({
    expertId,
    enabled,
  });

  if (items.length === 0) return null;

  return (
    <HomeTile
      icon={AlertCircleIcon}
      title="Needs you"
      badge={
        <Text
          variant="small-medium"
          as="span"
          tone="secondary"
          role="status"
          aria-label={`${items.length} ${items.length === 1 ? "item needs" : "items need"} your attention`}
          className="rounded-md bg-zinc-100 px-1.5 py-0.5 tabular-nums"
        >
          {items.length}
        </Text>
      }
    >
      <div className="divide-y divide-zinc-100">
        {items.map((item) => (
          <AttentionRow
            key={item.id}
            item={item}
            isProcessing={pendingIDs.has(item.id)}
            onDecision={decide}
          />
        ))}
      </div>
    </HomeTile>
  );
}
