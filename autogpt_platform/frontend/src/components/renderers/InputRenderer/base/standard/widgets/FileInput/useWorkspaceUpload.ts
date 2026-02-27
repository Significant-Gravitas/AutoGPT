import { useState } from "react";
import {
  uploadFileToWorkspace,
  WorkspaceUploadResult,
} from "@/services/workspace-upload";

export function useWorkspaceUpload() {
  const [uploadProgress, setUploadProgress] = useState(0);

  async function handleUploadFile(file: File): Promise<WorkspaceUploadResult> {
    setUploadProgress(0);
    const result = await uploadFileToWorkspace(file, setUploadProgress);
    setUploadProgress(100);
    return result;
  }

  return { handleUploadFile, uploadProgress };
}
