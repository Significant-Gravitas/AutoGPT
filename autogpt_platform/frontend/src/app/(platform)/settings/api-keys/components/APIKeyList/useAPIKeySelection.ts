import { useEffect, useState } from "react";

export function useAPIKeySelection(allIds: string[]) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    setSelectedIds((prev) => {
      if (prev.size === 0) return prev;
      const existing = new Set(allIds);
      const next = new Set<string>();
      for (const id of prev) {
        if (existing.has(id)) next.add(id);
      }
      return next.size === prev.size ? prev : next;
    });
  }, [allIds]);

  function toggle(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function selectAll() {
    setSelectedIds(new Set(allIds));
  }

  function clear() {
    setSelectedIds(new Set());
  }

  return {
    selectedIds,
    selectedCount: selectedIds.size,
    allSelected: allIds.length > 0 && selectedIds.size === allIds.length,
    isSelected: (id: string) => selectedIds.has(id),
    toggle,
    selectAll,
    clear,
  };
}
