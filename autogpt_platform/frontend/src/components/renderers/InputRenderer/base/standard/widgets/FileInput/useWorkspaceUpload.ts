import {
  usePostWorkspaceUploadAFileToTheWorkspace,
  useDeleteWorkspaceDeleteAWorkspaceFile,
} from "@/app/api/__generated__/endpoints/workspace/workspace";

export function parseWorkspaceFileID(uri: string): string | null {
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
    const d = response.data;
    return {
      file_name: d.name,
      size: d.size_bytes,
      content_type: d.mime_type,
      file_uri: `workspace://${d.file_id}#${d.mime_type}`,
    };
  }

  function handleDeleteFile(fileURI: string) {
    const fileID = parseWorkspaceFileID(fileURI);
    if (!fileID) return;
    deleteMutation({ fileId: fileID });
  }

  return { handleUploadFile, handleDeleteFile };
}
