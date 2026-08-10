import { AlertCircleIcon } from "@hugeicons/core-free-icons";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { HomeTileExpandButton } from "../../HomeTileExpandButton/HomeTileExpandButton";
import { HomeTileFilter } from "../../HomeTileFilter/HomeTileFilter";

type AttentionFilter = "all" | HomeAttentionItem["kind"];

interface Props {
  itemCount: number;
  hasFilters: boolean;
  selectedKind: AttentionFilter;
  filterOptions: { value: string; label: string }[];
  onSelectKind: (kind: AttentionFilter) => void;
  showAll: boolean;
  onToggleShowAll: () => void;
}

export function NeedsYouTitle({
  itemCount,
  hasFilters,
  selectedKind,
  filterOptions,
  onSelectKind,
  showAll,
  onToggleShowAll,
}: Props) {
  return (
    <div className="flex items-start justify-between gap-3">
      <div className="flex min-w-0 items-center gap-2">
        <Icon
          icon={AlertCircleIcon}
          size={18}
          className="text-zinc-500"
          aria-hidden="true"
        />
        <Text variant="h5" className="text-zinc-950">
          Needs you
        </Text>
      </div>
      <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
        {itemCount > 0 ? (
          <span
            role="status"
            aria-label={`${itemCount} ${itemCount === 1 ? "item needs" : "items need"} your attention`}
            className="flex size-6 items-center justify-center rounded-full bg-zinc-900 text-xs font-semibold tabular-nums text-white"
          >
            {itemCount}
          </span>
        ) : null}
        {hasFilters ? (
          <HomeTileFilter
            ariaLabelPrefix="Filter interventions"
            value={selectedKind}
            options={filterOptions}
            onChange={(value) => onSelectKind(value as AttentionFilter)}
          />
        ) : null}
        {itemCount > 0 ? (
          <HomeTileExpandButton
            label={
              showAll ? "Show fewer attention items" : "Expand attention items"
            }
            pressed={showAll}
            onClick={onToggleShowAll}
          />
        ) : null}
      </div>
    </div>
  );
}
