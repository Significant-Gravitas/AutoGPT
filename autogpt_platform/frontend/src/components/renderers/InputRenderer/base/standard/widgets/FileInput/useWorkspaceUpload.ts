import {
  usePostWorkspaceUploadFileToWorkspace,
  useDeleteWorkspaceDeleteAWorkspaceFile,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { parseWorkspaceFileID, buildWorkspaceURI } from "@/lib/workspace-uri";

export function useWorkspaceUpload() {
  const { toast } = useToast();

  const { mutateAsync: uploadMutation } =
    usePostWorkspaceUploadFileToWorkspace();

  const { mutate: deleteMutation } = useDeleteWorkspaceDeleteAWorkspaceFile({
    mutation: {
      onError: () => {
        toast({
          title: "Failed to delete file",
          description: "The file could not be removed from storage.",
          variant: "destructive",
        });
      },
    },
  });

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
      file_uri: buildWorkspaceURI(d.file_id, d.mime_type),
    };
  }

  function handleDeleteFile(fileURI: string) {
    const fileID = parseWorkspaceFileID(fileURI);
    if (!fileID) return;
    deleteMutation({ fileId: fileID });
  }

  return { handleUploadFile, handleDeleteFile };
}
