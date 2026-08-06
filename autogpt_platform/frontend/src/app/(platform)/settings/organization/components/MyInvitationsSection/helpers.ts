import { getV2ListUserOrganizations } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import type { UserInvitationResponse } from "@/app/api/__generated__/models/userInvitationResponse";
import { normalizeOrg } from "@/services/org-team/normalize";
import type { Org } from "@/services/org-team/store";

/**
 * OrgTeamProvider only loads the org list on auth changes, so an org joined
 * mid-session never lands in the store. Switching to an org that isn't in the
 * list leaves `activeOrg` null everywhere the switcher reads it, so pull the
 * authoritative list before switching — falling back to an entry built from
 * the invitation if that request fails, so the store stays consistent either
 * way (the provider refreshes the placeholder's counts on the next load).
 */
export async function getOrgsAfterJoin(
  invitation: UserInvitationResponse,
  currentOrgs: Org[],
): Promise<Org[]> {
  try {
    const response = await getV2ListUserOrganizations();
    return (response.data as OrgResponse[]).map(normalizeOrg);
  } catch {
    if (currentOrgs.some((org) => org.id === invitation.org_id)) {
      return currentOrgs;
    }
    return [
      ...currentOrgs,
      {
        id: invitation.org_id,
        name: invitation.org_name,
        slug: invitation.org_slug,
        avatarUrl: null,
        isPersonal: false,
        memberCount: 0,
      },
    ];
  }
}
