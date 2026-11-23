import { Directory, File, Paths } from "expo-file-system";

// Copies of workspace files made for the share sheet, kept apart so sign-out
// can wipe them in one go.
function folder() {
  return new Directory(Paths.cache, "shared-files");
}

/** An empty folder holding nothing but the file about to be shared. */
export function freshShareTarget(fileID: string, name: string) {
  clearSharedFiles();
  const dir = folder();
  dir.create({ intermediates: true, idempotent: true });
  // The id keeps two files of the same name apart; the name is what the other app shows.
  return new File(dir, `${fileID}-${name.replace(/[/\\]/g, "_")}`);
}

/** Deletes every shared copy. Called after a share and at sign-out. */
export function clearSharedFiles() {
  try {
    const dir = folder();
    if (dir.exists) dir.delete();
  } catch {
    // Nothing to remove, or the cache is already being cleared by the system.
  }
}
