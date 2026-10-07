"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { isHitSelected } from "./helpers";
import { KitResultRow } from "./KitResultRow";
import { KitSearchSkeleton } from "./KitSearchSkeleton";
import type { useAttachmentPicker } from "./useAttachmentPicker";

interface Props {
  picker: ReturnType<typeof useAttachmentPicker>;
  emptyQueryHint: string;
  emptyResultsHint: string;
}

export function KitSearchResults({
  picker,
  emptyQueryHint,
  emptyResultsHint,
}: Props) {
  if (picker.isSearching) {
    return <KitSearchSkeleton />;
  }

  if (picker.hits.length === 0) {
    return (
      <div className="flex w-full max-w-2xl animate-in flex-col items-center gap-2 rounded-2xl border border-dashed border-border px-6 py-8 text-center duration-300 fade-in motion-reduce:animate-none">
        <Icon
          icon={Search01Icon}
          size={20}
          aria-hidden
          className="text-muted-foreground/70"
        />
        <Text variant="body" tone="muted" className="max-w-96">
          {picker.hasQuery ? emptyResultsHint : emptyQueryHint}
        </Text>
      </div>
    );
  }

  return (
    <div
      role="list"
      aria-label="Search results"
      className="w-full max-w-2xl overflow-hidden rounded-2xl border border-border bg-background shadow-xs"
    >
      {picker.hits.map((hit, index) => (
        <KitResultRow
          key={hit.key}
          hit={hit}
          index={index}
          selected={isHitSelected(picker.attachments, hit)}
          atCap={picker.atCap}
          isPending={picker.pendingKey === hit.key}
          onAdd={() => picker.addHit(hit)}
        />
      ))}
    </div>
  );
}
