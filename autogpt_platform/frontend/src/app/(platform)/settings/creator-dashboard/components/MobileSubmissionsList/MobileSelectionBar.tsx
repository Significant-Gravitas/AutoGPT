"use client";

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

export function MobileSelectionBar({
  selectedCount,
  allSelected,
  onSelectAll,
  onDeselectAll,
  onDeleteSelected,
}: Props) {
  return (
    <div className="flex w-full min-w-0 max-w-full flex-col gap-2 rounded-[14px] border border-zinc-200 bg-zinc-100 px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <Text variant="body-medium" as="span" className="text-zinc-700">
          {selectedCount} selected
        </Text>
        <Button
          variant="destructive"
          size="small"
          leftIcon={<TrashIcon size={14} />}
          onClick={onDeleteSelected}
        >
          Delete
        </Button>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {!allSelected && (
          <Button variant="ghost" size="small" onClick={onSelectAll}>
            Select all
          </Button>
        )}
        <Button variant="ghost" size="small" onClick={onDeselectAll}>
          Deselect
        </Button>
      </div>
    </div>
  );
}
