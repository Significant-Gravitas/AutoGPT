import { postV1UploadFileToCloudStorage } from "@/app/api/__generated__/endpoints/files/files";
import { resolveResponse } from "@/app/api/helpers";
import { useState } from "react";
import { useOrgTeamStore } from "@/services/org-team/store";
import { getTenantRequestInit } from "@/components/contextual/TeamPicker/helpers";

interface RunAgentInputScope {
  organizationId: string | null;
  teamId: string | null;
}

export function useRunAgentInputs(scope?: RunAgentInputScope) {
  const [uploadProgress, setUploadProgress] = useState(0);
  const activeOrgID = useOrgTeamStore((state) => state.activeOrgID);
  const activeTeamID = useOrgTeamStore((state) => state.activeTeamID);
  const organizationId = scope ? scope.organizationId : activeOrgID;
  const teamId = scope ? scope.teamId : activeTeamID;

  async function handleUploadFile(file: File) {
    setUploadProgress(0);
    const result = await resolveResponse(
      postV1UploadFileToCloudStorage(
        { file },
        { expiration_hours: 24 },
        getTenantRequestInit(organizationId, teamId),
      ),
    );
    setUploadProgress(100);
    return result;
  }

  return {
    uploadProgress,
    handleUploadFile,
  };
}
