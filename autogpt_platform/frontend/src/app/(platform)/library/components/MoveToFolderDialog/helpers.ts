import type { LibraryFolderTree } from "@/app/api/__generated__/models/libraryFolderTree";

interface FolderAtDepth {
  folder: LibraryFolderTree;
  depth: number;
}

/** Every folder in the tree, each one followed by its subfolders. */
export function flattenFolderTree(
  tree: LibraryFolderTree[],
  depth = 0,
): FolderAtDepth[] {
  return tree.flatMap((folder) => [
    { folder, depth },
    ...flattenFolderTree(folder.children ?? [], depth + 1),
  ]);
}
