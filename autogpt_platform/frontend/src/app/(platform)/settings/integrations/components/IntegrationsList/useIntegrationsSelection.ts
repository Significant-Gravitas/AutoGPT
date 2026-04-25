"use client";

import { useEffect, useState } from "react";

type SelectionMap = Record<string, true>;

export function useIntegrationsSelection(allIds: string[]) {
  const [selected, setSelected] = useState<SelectionMap>({});

  useEffect(() => {
    setSelected((prev) => {
      const prevKeys = Object.keys(prev);
      if (prevKeys.length === 0) return prev;
      const existing = new Set(allIds);
      const next: SelectionMap = {};
      let removed = false;
      for (const id of prevKeys) {
        if (existing.has(id)) next[id] = true;
        else removed = true;
      }
      return removed ? next : prev;
    });
  }, [allIds]);

  function toggle(id: string) {
    setSelected((prev) => {
      if (prev[id]) {
        const next = { ...prev };
        delete next[id];
        return next;
      }
      return { ...prev, [id]: true };
    });
  }

  function selectAll() {
    const next: SelectionMap = {};
    for (const id of allIds) next[id] = true;
    setSelected(next);
  }

  function clear() {
    setSelected({});
  }

  const selectedIds = Object.keys(selected);
  const selectedCount = selectedIds.length;

  return {
    selectedIds,
    selectedCount,
    allSelected: allIds.length > 0 && selectedCount === allIds.length,
    isSelected: (id: string) => Boolean(selected[id]),
    toggle,
    selectAll,
    clear,
  };
}
