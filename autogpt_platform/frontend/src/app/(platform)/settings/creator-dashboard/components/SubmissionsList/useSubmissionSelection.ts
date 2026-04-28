import { useEffect, useState } from "react";

export function useSubmissionSelection(selectableIds: string[]) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    setSelectedIds((prev) => {
      if (prev.size === 0) return prev;
      const existing = new Set(selectableIds);
      const next = new Set<string>();
      for (const id of prev) {
        if (existing.has(id)) next.add(id);
      }
      return next.size === prev.size ? prev : next;
    });
  }, [selectableIds]);

  function toggle(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function selectAll() {
    setSelectedIds(new Set(selectableIds));
  }

  function clear() {
    setSelectedIds(new Set());
  }

  return {
    selectedIds,
    selectedCount: selectedIds.size,
    allSelected:
      selectableIds.length > 0 &&
      selectableIds.every((id) => selectedIds.has(id)),
    isSelected: (id: string) => selectedIds.has(id),
    toggle,
    selectAll,
    clear,
  };
}
