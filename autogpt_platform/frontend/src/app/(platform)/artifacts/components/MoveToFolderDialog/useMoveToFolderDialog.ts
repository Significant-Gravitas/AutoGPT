import { useState } from "react";
import { useArtifactsFolders } from "../../useArtifactsFolders";
import {
  buildTreeRows,
  initiallyExpanded,
  rowKey,
  type MoveSubject,
} from "./helpers";

interface Args {
  move: MoveSubject;
  canMoveToRoot: boolean;
  onDone: () => void;
}

export function useMoveToFolderDialog({ move, canMoveToRoot, onDone }: Args) {
  const { folders, moveFilesToFolder, moveFolder, isMovingFolder } =
    useArtifactsFolders();
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  // Seeded once from the subject's own location, then owned by the user; a
  // later folders refetch must not re-collapse what they opened.
  const [expanded, setExpanded] = useState(() =>
    initiallyExpanded(folders, move),
  );
  const [isMovingFiles, setIsMovingFiles] = useState(false);

  const rows = buildTreeRows({ folders, move, expanded, canMoveToRoot });

  function toggleExpanded(folderId: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(folderId)) next.delete(folderId);
      else next.add(folderId);
      return next;
    });
  }

  function confirm() {
    const target = rows.find((row) => rowKey(row.id) === selectedKey);
    if (!target || target.disabledReason) return;
    // Closes only on success; the hook toasts on error and the dialog stays
    // open so the user can pick another destination without reopening it.
    if (move.kind === "folder") {
      const name = folders.find((f) => f.id === move.folderId)?.name ?? "";
      moveFolder({ folderId: move.folderId, parentId: target.id, name })
        .then(onDone)
        .catch(() => {});
      return;
    }
    setIsMovingFiles(true);
    moveFilesToFolder({ fileIds: move.fileIds, folderId: target.id })
      .then(onDone)
      .catch(() => {})
      .finally(() => setIsMovingFiles(false));
  }

  return {
    rows,
    selectedKey,
    select: (id: string | null) => setSelectedKey(rowKey(id)),
    toggleExpanded,
    confirm,
    canConfirm: selectedKey !== null,
    isMoving: isMovingFolder || isMovingFiles,
  };
}
