import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";

export type SelectionModifiers = { shift?: boolean; meta?: boolean };

export interface Selection {
  selected: ReadonlyMap<string, WorkspaceFileItem>;
  /** Id of the last plainly clicked file. An id, not an index, because a
   *  refetch can reorder the list under it. */
  anchor: string | null;
}

/**
 * Applies one click to the selection, following the range conventions people
 * already expect from a file manager: plain click toggles and re-anchors,
 * Shift extends from the anchor, Ctrl/Cmd toggles without moving it.
 */
export function applySelection(
  current: Selection,
  files: WorkspaceFileItem[],
  index: number,
  modifiers: SelectionModifiers = {},
): Selection {
  const file = files[index];
  if (!file) return current;

  // Shift with no anchor in the list has nothing to extend from, so it
  // behaves as the plain click that establishes one.
  const anchorIndex = files.findIndex((f) => f.id === current.anchor);
  if (modifiers.shift && anchorIndex !== -1) {
    return {
      selected: addRange(current.selected, files, anchorIndex, index),
      anchor: current.anchor,
    };
  }
  const selected = toggleOne(current.selected, file);
  return { selected, anchor: modifiers.meta ? current.anchor : file.id };
}

/**
 * Selection order is insertion order, which for a backwards range is the
 * order clicked rather than the order shown. Sort by the list so the chips —
 * and, at the cap, the files that survive it — match what the user saw.
 * Anything no longer in `files` (paged off, or filtered out) keeps its place
 * at the end.
 */
export function orderByList(
  selected: ReadonlyMap<string, WorkspaceFileItem>,
  files: WorkspaceFileItem[],
): WorkspaceFileItem[] {
  const listed: WorkspaceFileItem[] = [];
  const unlisted = new Map(selected);
  for (const file of files) {
    if (unlisted.delete(file.id)) listed.push(file);
  }
  return [...listed, ...unlisted.values()];
}

/** A range only ever adds, so dragging back over chosen files cannot clear
 *  them — the union is what makes repeated ranges additive. */
function addRange(
  selected: ReadonlyMap<string, WorkspaceFileItem>,
  files: WorkspaceFileItem[],
  anchor: number,
  index: number,
): Map<string, WorkspaceFileItem> {
  const next = new Map(selected);
  const from = Math.min(anchor, index);
  const to = Math.max(anchor, index);
  for (const file of files.slice(from, to + 1)) {
    next.set(file.id, file);
  }
  return next;
}

function toggleOne(
  selected: ReadonlyMap<string, WorkspaceFileItem>,
  file: WorkspaceFileItem,
): Map<string, WorkspaceFileItem> {
  const next = new Map(selected);
  if (next.has(file.id)) {
    next.delete(file.id);
  } else {
    next.set(file.id, file);
  }
  return next;
}
