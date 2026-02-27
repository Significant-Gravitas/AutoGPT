export interface WorkspaceUploadResult {
  file_uri: string;
  file_name: string;
  size: number;
  content_type: string;
}

/**
 * Upload a file to the workspace via XHR with progress tracking.
 * Posts to the Next.js proxy which injects auth and forwards to the backend.
 */
export function uploadFileToWorkspace(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<WorkspaceUploadResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file);

    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress((event.loaded / event.total) * 100);
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const result: WorkspaceUploadResult = JSON.parse(xhr.responseText);
          resolve(result);
        } catch {
          reject(new Error("Invalid response from server"));
        }
      } else {
        let message = `Upload failed (${xhr.status})`;
        try {
          const err = JSON.parse(xhr.responseText);
          if (err.detail) message = err.detail;
        } catch {}
        reject(new Error(message));
      }
    });

    xhr.addEventListener("error", () => {
      reject(new Error("Network error during upload"));
    });

    xhr.addEventListener("abort", () => {
      reject(new Error("Upload aborted"));
    });

    xhr.open("POST", "/api/proxy/api/workspace/files/upload");
    xhr.send(formData);
  });
}
