import {
  usePostWorkspaceUploadAFileToTheWorkspace,
  useDeleteWorkspaceDeleteAWorkspaceFile,
} from "@/app/api/__generated__/endpoints/workspace/workspace";

function parseWorkspaceFileID(uri: string): string | null {
  if (!uri.startsWith("workspace://")) return null;
  const rest = uri.slice("workspace://".length);
  const hashIndex = rest.indexOf("#");
  return hashIndex === -1 ? rest : rest.slice(0, hashIndex);
}

export function useWorkspaceUpload() {
  const { mutateAsync: uploadMutation } =
    usePostWorkspaceUploadAFileToTheWorkspace();

  const { mutate: deleteMutation } = useDeleteWorkspaceDeleteAWorkspaceFile();

  async function handleUploadFile(file: File) {
    const response = await uploadMutation({ data: { file } });
    if (response.status !== 200) {
      throw new Error("Upload failed");
    }
    return response.data;
  }

  function handleDeleteFile(fileURI: string) {
    const fileID = parseWorkspaceFileID(fileURI);
    if (!fileID) return;
    deleteMutation({ fileId: fileID });
  }

  return { handleUploadFile, handleDeleteFile };
}
