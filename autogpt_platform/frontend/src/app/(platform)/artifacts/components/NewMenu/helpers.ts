import { bulkMoveWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { getTenantRequestInit } from "@/components/contextual/TeamPicker/helpers";
import type { UploadTenantScope } from "@/lib/direct-upload";
import { uploadFileDirect } from "@/lib/direct-upload";

export interface UploadOutcome {
  uploaded: number;
  // Uploaded fine, but the move into the selected folder failed, so the file
  // sits at the workspace root.
  leftAtRoot: string[];
  failed: { name: string; message: string }[];
}

// The upload endpoint has no folder parameter: every upload lands at the root
// and is moved into the selected folder as a second step.
export async function uploadFiles(
  files: File[],
  folderId: string | null,
  scope?: UploadTenantScope,
): Promise<UploadOutcome> {
  const outcome: UploadOutcome = { uploaded: 0, leftAtRoot: [], failed: [] };
  for (const file of files) {
    let fileId: string;
    try {
      fileId = (await uploadFileDirect(file, undefined, scope)).file_id;
    } catch (error) {
      outcome.failed.push({
        name: file.name,
        message: error instanceof Error ? error.message : "Please try again.",
      });
      continue;
    }
    outcome.uploaded += 1;
    if (folderId && !(await moveIntoFolder(fileId, folderId, scope))) {
      outcome.leftAtRoot.push(file.name);
    }
  }
  return outcome;
}

async function moveIntoFolder(
  fileId: string,
  folderId: string,
  scope?: UploadTenantScope,
): Promise<boolean> {
  try {
    const res = await bulkMoveWorkspaceFiles(
      {
        file_ids: [fileId],
        folder_id: folderId,
      },
      scope
        ? getTenantRequestInit(scope.organizationId, scope.teamId)
        : undefined,
    );
    return res.status === 200;
  } catch {
    return false;
  }
}
