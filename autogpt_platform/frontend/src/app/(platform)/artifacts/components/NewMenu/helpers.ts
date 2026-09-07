import { bulkMoveWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { uploadFileDirect } from "@/lib/direct-upload";

export interface UploadOutcome {
  uploaded: number;
  failed: { name: string; message: string }[];
}

// Uploads land at the workspace root, so a file added while browsing a folder
// is moved into that folder right after it lands.
export async function uploadFiles(
  files: File[],
  folderId: string | null,
): Promise<UploadOutcome> {
  const outcome: UploadOutcome = { uploaded: 0, failed: [] };
  for (const file of files) {
    try {
      const result = await uploadFileDirect(file);
      if (folderId) {
        const moved = await bulkMoveWorkspaceFiles({
          file_ids: [result.file_id],
          folder_id: folderId,
        });
        if (moved.status !== 200) {
          throw new Error("Uploaded to the root, but couldn't move it here.");
        }
      }
      outcome.uploaded += 1;
    } catch (error) {
      outcome.failed.push({
        name: file.name,
        message: error instanceof Error ? error.message : "Please try again.",
      });
    }
  }
  return outcome;
}
