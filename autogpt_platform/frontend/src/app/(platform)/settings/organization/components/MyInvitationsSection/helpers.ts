import { getV2ListUserOrganizations } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import type { UserInvitationResponse } from "@/app/api/__generated__/models/userInvitationResponse";
import { normalizeOrg } from "@/services/org-team/normalize";
import type { Org } from "@/services/org-team/store";

/**
 * OrgTeamProvider only loads the org list on auth changes, so an org joined
 * mid-session never lands in the store. Switching to an org that isn't in the
 * list leaves `activeOrg` null everywhere the switcher reads it, so pull the
 * authoritative list before switching.
 *
 * The joined org is guaranteed to come back in the result whatever the refresh
 * does — request failed, or succeeded but hasn't caught up with the membership
 * yet — by appending an entry derived from the invitation. The provider
 * refreshes that placeholder's avatar and member count on its next load.
 */
export async function getOrgsAfterJoin(
  invitation: UserInvitationResponse,
  currentOrgs: Org[],
): Promise<Org[]> {
  let orgs = currentOrgs;

  try {
    const response = await getV2ListUserOrganizations();
    orgs = (response.data as OrgResponse[]).map(normalizeOrg);
  } catch {
    // Keep the list we already have and fall through to the placeholder.
  }

  if (orgs.some((org) => org.id === invitation.org_id)) {
    return orgs;
  }

  return [
    ...orgs,
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
