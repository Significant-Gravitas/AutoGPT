import { useGetV2ListOrganizationMembers } from "@/app/api/__generated__/endpoints/orgs/orgs";
import { okData } from "@/app/api/helpers";
import { getTenantRequestInit } from "@/components/contextual/TeamPicker/helpers";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useOrgTeamStore } from "@/services/org-team/store";

export function usePersonalExpertWorkspace() {
  const { user } = useAuth();
  const { activeOrgID, activeTeamID, orgs, teams, isLoaded } =
    useOrgTeamStore();
  const org = orgs.find((candidate) => candidate.id === activeOrgID);
  const needsMembership = !!activeOrgID && !!org?.isPersonal;
  const members = useGetV2ListOrganizationMembers(activeOrgID ?? "", {
    query: {
      enabled: isLoaded && needsMembership && !!user?.id,
      select: (response) => okData(response) ?? [],
    },
    request: getTenantRequestInit(activeOrgID, null, isLoaded),
  });
  const isOwner =
    members.data?.some(
      (member) => member.user_id === user?.id && member.is_owner,
    ) ?? false;
  const isDefaultTeam =
    activeTeamID === null ||
    teams.some((team) => team.id === activeTeamID && team.isDefault);
  const isReady =
    isLoaded && (!needsMembership || !user?.id || !members.isPending);
  const canUseExperts =
    isReady &&
    isDefaultTeam &&
    (!activeOrgID || (!!org?.isPersonal && isOwner));
  return {
    canUseExperts,
    isReady,
    request: getTenantRequestInit(activeOrgID, activeTeamID, isReady),
  };
}
