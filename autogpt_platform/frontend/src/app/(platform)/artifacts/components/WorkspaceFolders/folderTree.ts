import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";

/**
 * Nesting read off the flat `GET /folders` list. Every walk is bounded by the
 * number of folders, so a `parent_id` cycle the API should never produce stops
 * instead of hanging the render.
 */

export function childrenOf(
  folders: WorkspaceFolder[],
  parentId: string | null,
): WorkspaceFolder[] {
  return folders
    .filter((folder) => (folder.parent_id ?? null) === parentId)
    .sort((a, b) => a.name.localeCompare(b.name));
}

/** Root first, the folder itself last. Empty when the id is unknown. */
export function ancestorsOf(
  folders: WorkspaceFolder[],
  folderId: string,
): WorkspaceFolder[] {
  const byId = indexById(folders);
  const chain: WorkspaceFolder[] = [];
  const seen = new Set<string>();
  let current = byId.get(folderId);
  while (current && !seen.has(current.id)) {
    seen.add(current.id);
    chain.push(current);
    const parentId = current.parent_id ?? null;
    current = parentId ? byId.get(parentId) : undefined;
  }
  return chain.reverse();
}

/** Every folder under `folderId`, excluding the folder itself. */
export function descendantIdsOf(
  folders: WorkspaceFolder[],
  folderId: string,
): Set<string> {
  const byParent = indexByParent(folders);
  const found = new Set<string>();
  const queue = [folderId];
  while (queue.length > 0) {
    const id = queue.shift() as string;
    for (const child of byParent.get(id) ?? []) {
      if (found.has(child.id)) continue;
      found.add(child.id);
      queue.push(child.id);
    }
  }
  return found;
}

export function subfolderCountOf(
  folders: WorkspaceFolder[],
  folderId: string,
): number {
  return folders.filter((folder) => (folder.parent_id ?? null) === folderId)
    .length;
}

/**
 * The one-line summary a folder gets wherever it is not being browsed —
 * direct counts only, so it never claims a subtree total it did not count.
 */
export function folderSummary(fileCount: number, subfolderCount: number) {
  if (fileCount > 0)
    return `${fileCount} ${fileCount === 1 ? "file" : "files"}`;
  if (subfolderCount > 0)
    return `${subfolderCount} ${subfolderCount === 1 ? "folder" : "folders"}`;
  return "Empty";
}

function indexById(folders: WorkspaceFolder[]) {
  return new Map(folders.map((folder) => [folder.id, folder]));
}

function indexByParent(folders: WorkspaceFolder[]) {
  const byParent = new Map<string, WorkspaceFolder[]>();
  for (const folder of folders) {
    const parentId = folder.parent_id ?? null;
    if (parentId === null) continue;
    const siblings = byParent.get(parentId);
    if (siblings) siblings.push(folder);
    else byParent.set(parentId, [folder]);
  }
  return byParent;
}
