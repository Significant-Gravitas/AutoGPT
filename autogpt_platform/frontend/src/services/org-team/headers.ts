import { useOrgTeamStore } from "./store";

export { ORG_HEADER_NAME, TEAM_HEADER_NAME } from "./header-names";
import { ORG_HEADER_NAME, TEAM_HEADER_NAME } from "./header-names";

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
