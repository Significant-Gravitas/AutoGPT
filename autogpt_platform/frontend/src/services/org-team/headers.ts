import { useOrgTeamStore } from "./store";

export const ORG_HEADER_NAME = "X-Org-Id";
export const TEAM_HEADER_NAME = "X-Team-Id";

export function getOrgContextHeaders(): Record<string, string> {
  const { activeOrgID, activeTeamID } = useOrgTeamStore.getState();
  const headers: Record<string, string> = {};
  if (activeOrgID) {
    headers[ORG_HEADER_NAME] = activeOrgID;
    if (activeTeamID) {
      headers[TEAM_HEADER_NAME] = activeTeamID;
    }
  }
  return headers;
}
