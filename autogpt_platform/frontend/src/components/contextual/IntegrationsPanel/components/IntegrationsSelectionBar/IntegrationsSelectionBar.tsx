"use client";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
    <div className="flex w-full items-center justify-between rounded-sm border border-border-base bg-surface-hover px-4 py-2">
      <div className="flex items-center gap-5">
        <Text variant="body" as="span" tone="body">
          {selectedCount} selected
        </Text>
        {!allSelected && (
          <button
            type="button"
            onClick={onSelectAll}
            className="rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-action"
          >
            <Text variant="body-medium" as="span">
              Select All
            </Text>
          </button>
        )}
        <button
          type="button"
          onClick={onDeselectAll}
          className="rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-action"
        >
          <Text variant="body-medium" as="span">
            Deselect
          </Text>
        </button>
      </div>
      <Button
        variant="destructive"
        size="small"
        leftIcon={<Icon icon={Delete02Icon} size={16} />}
        onClick={onDeleteSelected}
        loading={isDeleting}
        disabled={isDeleting}
      >
        Delete selected
      </Button>
    </div>
  );
}
