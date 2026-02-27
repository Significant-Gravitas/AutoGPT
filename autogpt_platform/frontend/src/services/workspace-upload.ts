export interface WorkspaceUploadResult {
  file_uri: string;
  file_name: string;
  size: number;
  content_type: string;
}

/**
 * Extract the file ID from a workspace:// URI.
 * Returns null if the value is not a workspace URI.
 */
export function parseWorkspaceFileID(uri: string): string | null {
  if (!uri.startsWith("workspace://")) return null;
  const rest = uri.slice("workspace://".length);
  const hashIndex = rest.indexOf("#");
  return hashIndex === -1 ? rest : rest.slice(0, hashIndex);
}

/**
 * Upload a file to the workspace via the Next.js proxy.
 */
export async function uploadWorkspaceFile(
  file: File,
): Promise<WorkspaceUploadResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/proxy/api/workspace/files/upload", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = `Upload failed (${response.status})`;
    try {
      const err = await response.json();
      if (err.detail) message = err.detail;
    } catch {}
    throw new Error(message);
  }

  return response.json();
}

/**
 * Delete a workspace file by its ID via the Next.js proxy.
 */
export async function deleteWorkspaceFile(fileID: string): Promise<void> {
  const response = await fetch(`/api/proxy/api/workspace/files/${fileID}`, {
    method: "DELETE",
  });
  if (!response.ok && response.status !== 404) {
    throw new Error(`Failed to delete workspace file (${response.status})`);
  }
}
