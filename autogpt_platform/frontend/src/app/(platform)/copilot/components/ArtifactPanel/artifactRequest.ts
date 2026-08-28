import { getCopilotAuthHeaders } from "../../helpers";
import type { ArtifactRef } from "../../store";
import { getTenantRequestInit } from "@/components/contextual/TeamPicker/helpers";

function getArtifactScope(artifact: ArtifactRef) {
  if (!("organizationId" in artifact) && !("teamId" in artifact)) {
    return undefined;
  }
  return {
    organizationId: artifact.organizationId ?? null,
    teamId: artifact.teamId ?? null,
  };
}

export async function fetchArtifactResource(
  artifact: ArtifactRef,
): Promise<Response> {
  const scope = getArtifactScope(artifact);
  if (!scope) return fetch(artifact.sourceUrl);
  if (artifact.sourceUrl.startsWith("/api/")) {
    return fetch(
      artifact.sourceUrl,
      getTenantRequestInit(scope.organizationId, scope.teamId),
    );
  }
  return fetch(artifact.sourceUrl, {
    headers: await getCopilotAuthHeaders(scope),
  });
}
