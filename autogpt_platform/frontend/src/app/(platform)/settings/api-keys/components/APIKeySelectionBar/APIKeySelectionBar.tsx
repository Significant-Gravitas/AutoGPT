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
}

export function APIKeySelectionBar({
  selectedCount,
  allSelected,
  onSelectAll,
  onDeselectAll,
  onDeleteSelected,
}: Props) {
  return (
    <div className="flex w-full items-center justify-between rounded-[4px] border border-border-base bg-surface-hover px-4 py-2">
      <div className="flex items-center gap-5">
        <Text variant="body" as="span" className="text-foreground-body">
          {selectedCount} selected
        </Text>
        {!allSelected && (
          <Button variant="ghost" size="small" onClick={onSelectAll}>
            Select All
          </Button>
        )}
        <Button variant="ghost" size="small" onClick={onDeselectAll}>
          Deselect
        </Button>
      </div>
      <Button
        variant="destructive"
        size="small"
        leftIcon={<Icon icon={Delete02Icon} size={16} />}
        onClick={onDeleteSelected}
      >
        Delete selected
      </Button>
    </div>
  );
}
