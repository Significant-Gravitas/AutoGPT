"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
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
      <div className="flex w-full max-w-[42rem] flex-col items-center gap-2 rounded-2xl border border-dashed border-border-base px-6 py-8 text-center duration-300 animate-in fade-in motion-reduce:animate-none">
        <Icon
          icon={Search01Icon}
          size={20}
          aria-hidden
          className="text-foreground-tertiary/70"
        />
        <p className="max-w-[24rem] text-body text-foreground-tertiary">
          {picker.hasQuery ? emptyResultsHint : emptyQueryHint}
        </p>
      </div>
    );
  }

  return (
    <div
      role="list"
      aria-label="Search results"
      className="w-full max-w-[42rem] overflow-hidden rounded-2xl border border-border-base bg-canvas shadow-sm"
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
