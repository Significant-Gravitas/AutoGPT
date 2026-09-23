import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import {
  ancestorsOf,
  childrenOf,
  descendantIdsOf,
} from "../WorkspaceFolders/folderTree";

/** What is being moved: a set of files, or one folder. */
export type MoveSubject =
  | { kind: "files"; fileIds: string[]; currentFolderId?: string | null }
  | { kind: "folder"; folderId: string };

export interface TreeRow {
  /** `null` is the workspace root. */
  id: string | null;
  name: string;
  level: number;
  hasChildren: boolean;
  isExpanded: boolean;
  /** Why this destination is refused, shown beside the name. */
  disabledReason: string | null;
}

/**
 * The rows the tree paints, in order, for the folders currently expanded.
 * Disabled rows stay in the list and stay expandable: a refused folder can
 * still hold a valid destination.
 */
export function buildTreeRows(args: {
  folders: WorkspaceFolder[];
  move: MoveSubject;
  expanded: ReadonlySet<string>;
  canMoveToRoot: boolean;
}): TreeRow[] {
  const { folders, move, expanded, canMoveToRoot } = args;
  const refused = refusedTargets(folders, move);
  const rows: TreeRow[] = [];

  if (canMoveToRoot) {
    rows.push({
      id: null,
      name: "Files (root)",
      level: 1,
      hasChildren: false,
      isExpanded: false,
      disabledReason: refused.get(ROOT_KEY) ?? null,
    });
  }

  function walk(parentId: string | null, level: number) {
    for (const folder of childrenOf(folders, parentId)) {
      const isExpanded = expanded.has(folder.id);
      const hasChildren = childrenOf(folders, folder.id).length > 0;
      rows.push({
        id: folder.id,
        name: folder.name,
        level,
        hasChildren,
        isExpanded: hasChildren && isExpanded,
        disabledReason: refused.get(folder.id) ?? null,
      });
      if (hasChildren && isExpanded) walk(folder.id, level + 1);
    }
  }
  walk(null, 1);
  return rows;
}

/** Ancestors of where the subject sits now, so the tree opens showing it. */
export function initiallyExpanded(
  folders: WorkspaceFolder[],
  move: MoveSubject,
): Set<string> {
  const from =
    move.kind === "folder" ? move.folderId : (move.currentFolderId ?? null);
  if (!from) return new Set();
  return new Set(ancestorsOf(folders, from).map((folder) => folder.id));
}

export function rowKey(id: string | null): string {
  return id ?? ROOT_KEY;
}

/**
 * The destination a selection names, or null once it is refused or gone.
 * Resolved against every folder, not the visible rows: collapsing a selected
 * row's parent must not turn Move into a silent no-op.
 */
export function selectedTarget(args: {
  folders: WorkspaceFolder[];
  move: MoveSubject;
  canMoveToRoot: boolean;
  selectedKey: string | null;
}): { id: string | null } | null {
  const { folders, move, canMoveToRoot, selectedKey } = args;
  if (selectedKey === null) return null;
  if (refusedTargets(folders, move).has(selectedKey)) return null;
  if (selectedKey === ROOT_KEY) return canMoveToRoot ? { id: null } : null;
  return folders.some((f) => f.id === selectedKey) ? { id: selectedKey } : null;
}

const ROOT_KEY = "__root__";

function refusedTargets(
  folders: WorkspaceFolder[],
  move: MoveSubject,
): Map<string, string> {
  const refused = new Map<string, string>();
  if (move.kind === "folder") {
    const self = folders.find((folder) => folder.id === move.folderId);
    refused.set(move.folderId, "Folder being moved");
    for (const id of descendantIdsOf(folders, move.folderId)) {
      refused.set(id, "Inside the folder being moved");
    }
    refused.set(rowKey(self?.parent_id ?? null), "Current location");
    return refused;
  }
  // Null means "at the root" or "spread across folders", and only the first
  // is a location; `canMoveToRoot` already hides the root row in that case.
  if (move.currentFolderId) {
    refused.set(move.currentFolderId, "Current location");
  }
  return refused;
}
