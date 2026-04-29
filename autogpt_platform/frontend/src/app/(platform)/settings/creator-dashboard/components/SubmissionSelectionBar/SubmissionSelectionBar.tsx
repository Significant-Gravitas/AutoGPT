import { TrashIcon } from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  selectedCount: number;
  allSelected: boolean;
  onSelectAll: () => void;
  onDeselectAll: () => void;
  onDeleteSelected: () => void;
}

export function SubmissionSelectionBar({
  selectedCount,
  allSelected,
  onSelectAll,
  onDeselectAll,
  onDeleteSelected,
}: Props) {
  return (
    <div className="flex w-full items-center justify-between rounded-[14px] border border-zinc-200 bg-zinc-100 px-4 py-2">
      <div className="flex items-center gap-5">
        <Text variant="body" as="span" className="text-zinc-700">
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
        leftIcon={<TrashIcon size={16} />}
        onClick={onDeleteSelected}
      >
        Delete selected
      </Button>
    </div>
  );
}
