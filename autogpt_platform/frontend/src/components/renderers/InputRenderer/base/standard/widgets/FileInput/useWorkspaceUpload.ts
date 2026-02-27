import { useMutation } from "@tanstack/react-query";
import {
  uploadWorkspaceFile,
  deleteWorkspaceFile,
  parseWorkspaceFileID,
} from "@/services/workspace-upload";

export function useWorkspaceUpload() {
  const { mutateAsync: uploadFile } = useMutation({
    mutationFn: uploadWorkspaceFile,
  });

  const { mutate: deleteFile } = useMutation({
    mutationFn: deleteWorkspaceFile,
  });

  function handleDeleteFile(fileURI: string) {
    const fileID = parseWorkspaceFileID(fileURI);
    if (!fileID) return;
    deleteFile(fileID);
  }

  return { handleUploadFile: uploadFile, handleDeleteFile };
}
