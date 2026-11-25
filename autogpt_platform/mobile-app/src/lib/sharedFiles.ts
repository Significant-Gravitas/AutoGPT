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

// On iOS, expo-sharing links or copies a file here when its name lacks the
// extension its MIME type declares, and only clears it on a later share or launch.
function sharingStage() {
  return new Directory(Paths.cache, "expo-sharing-tmp");
}

/** Deletes every shared copy, expo-sharing's included. Called after a share and at sign-out. */
export function clearSharedFiles() {
  for (const dir of [folder(), sharingStage()]) {
    try {
      if (dir.exists) dir.delete();
    } catch {
      // Nothing to remove, or the cache is already being cleared by the system.
    }
  }
}
