import type { FolderUpdateRequest } from "@/app/api/__generated__/models/folderUpdateRequest";
import type { LibraryFolder } from "@/app/api/__generated__/models/libraryFolder";

export function applyFolderEdit(
  folder: LibraryFolder,
  edit: FolderUpdateRequest,
): LibraryFolder {
  return {
    ...folder,
    name: edit.name ?? folder.name,
    color: edit.color ?? folder.color,
    icon: edit.icon ?? folder.icon,
  };
}
