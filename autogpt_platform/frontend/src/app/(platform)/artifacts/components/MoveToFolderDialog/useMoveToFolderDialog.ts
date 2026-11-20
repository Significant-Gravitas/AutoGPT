import { useState } from "react";
import { useArtifactsFolders } from "../../useArtifactsFolders";
import {
  buildTreeRows,
  initiallyExpanded,
  rowKey,
  selectedTarget,
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
  // Seeded from the subject's own location, then owned by the user; a later
  // folders refetch must not re-collapse what they opened. The seed waits for
  // the folders query: on a cold cache the first render has nothing to walk,
  // and a lazy initialiser would leave the subject's own branch shut.
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(new Set());
  const [isSeeded, setIsSeeded] = useState(false);
  if (!isSeeded && folders.length > 0) {
    setIsSeeded(true);
    setExpanded(initiallyExpanded(folders, move));
  }
  const [isMovingFiles, setIsMovingFiles] = useState(false);

  const rows = buildTreeRows({ folders, move, expanded, canMoveToRoot });
  const target = selectedTarget({ folders, move, canMoveToRoot, selectedKey });

  function toggleExpanded(folderId: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(folderId)) next.delete(folderId);
      else next.add(folderId);
      return next;
    });
  }

  function confirm() {
    if (!target) return;
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
    canConfirm: target !== null,
    isMoving: isMovingFolder || isMovingFiles,
  };
}
