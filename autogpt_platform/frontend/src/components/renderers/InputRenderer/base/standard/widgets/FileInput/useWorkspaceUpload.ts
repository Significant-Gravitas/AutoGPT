import { useDeleteWorkspaceFile } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { uploadFileDirect } from "@/lib/direct-upload";
import { parseWorkspaceFileID, buildWorkspaceURI } from "@/lib/workspace-uri";
import { useOrgTeamStore } from "@/services/org-team/store";
import { getTenantRequestInit } from "@/components/contextual/TeamPicker/helpers";

interface WorkspaceUploadScope {
  organizationId: string | null;
  teamId: string | null;
}

export function useWorkspaceUpload(scope?: WorkspaceUploadScope) {
  const { toast } = useToast();
  const activeOrgID = useOrgTeamStore((s) => s.activeOrgID);
  const activeTeamID = useOrgTeamStore((s) => s.activeTeamID);
  const organizationId = scope ? scope.organizationId : activeOrgID;
  const teamId = scope ? scope.teamId : activeTeamID;

  const { mutate: deleteMutation } = useDeleteWorkspaceFile({
    request: getTenantRequestInit(organizationId, teamId),
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
    const d = await uploadFileDirect(file, undefined, {
      organizationId,
      teamId,
    });
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
