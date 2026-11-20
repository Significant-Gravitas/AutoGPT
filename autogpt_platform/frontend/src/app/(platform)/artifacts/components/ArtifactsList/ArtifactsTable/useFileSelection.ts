import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { useState } from "react";

/**
 * Multi-select for table rows. The selection is derived from the current
 * `files`, so an id that leaves the list (deleted, moved, paged out) drops out
 * of the selection on its own.
 */
export function useFileSelection(files: WorkspaceFileItem[]) {
  const [selectedIdSet, setSelectedIdSet] = useState<ReadonlySet<string>>(
    () => new Set(),
  );

  const selectedFiles = files.filter((file) => selectedIdSet.has(file.id));
  const selectedIds = selectedFiles.map((file) => file.id);

  function toggle(file: WorkspaceFileItem) {
    setSelectedIdSet((prev) => {
      const next = new Set(prev);
      if (next.has(file.id)) next.delete(file.id);
      else next.add(file.id);
      return next;
    });
  }

  function selectAll() {
    setSelectedIdSet(new Set(files.map((file) => file.id)));
  }

  function clear() {
    setSelectedIdSet(new Set());
  }

  return {
    selectedFiles,
    selectedIds,
    isSelected: (file: WorkspaceFileItem) => selectedIdSet.has(file.id),
    toggle,
    selectAll,
    clear,
  };
}
