"use client";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Delete02Icon } from "@hugeicons/core-free-icons";

interface Props {
  selectedCount: number;
  allSelected: boolean;
  onSelectAll: () => void;
  onDeselectAll: () => void;
  onDeleteSelected: () => void;
  isDeleting?: boolean;
}

export function IntegrationsSelectionBar({
  selectedCount,
  allSelected,
  onSelectAll,
  onDeselectAll,
  onDeleteSelected,
  isDeleting = false,
}: Props) {
  return (
    <div className="flex w-full items-center justify-between rounded border border-zinc-200 bg-zinc-100 px-4 py-2">
      <div className="flex items-center gap-5">
        <Text variant="body" as="span" tone="secondary">
          {selectedCount} selected
        </Text>
        {!allSelected && (
          <Button variant="link" onClick={onSelectAll}>
            Select All
          </Button>
        )}
        <Button variant="link" onClick={onDeselectAll}>
          Deselect
        </Button>
      </div>
      <Button
        variant="destructive"
        size="small"
        leadingIcon={Delete02Icon}
        onClick={onDeleteSelected}
        loading={isDeleting}
        disabled={isDeleting}
      >
        Delete selected
      </Button>
    </div>
  );
}
