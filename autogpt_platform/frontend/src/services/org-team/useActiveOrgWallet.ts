import { useOrgTeamStore } from "./store";

// Which wallet the credits screens read: the active team organization's, or,
// in a personal organization, the person's own.
export function useActiveOrgWallet() {
  const organizationID = useOrgTeamStore((state) => state.activeOrgID);
  const isTeamOrg = useOrgTeamStore((state) =>
    state.orgs.some((org) => org.id === state.activeOrgID && !org.isPersonal),
  );
  return { organizationID, isTeamOrg };
}
