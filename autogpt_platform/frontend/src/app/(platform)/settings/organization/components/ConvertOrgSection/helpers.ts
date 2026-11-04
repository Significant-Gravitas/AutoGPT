import { getV2ListUserOrganizations } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { normalizeOrg } from "@/services/org-team/normalize";
import type { Org } from "@/services/org-team/store";

export const CONVERT_CONSEQUENCES = [
  "Your workflows and store listings move into the new team organization.",
  "You get a fresh personal organization.",
  "The team organization starts with an empty wallet. It needs a team plan, set by an AutoGPT admin, before people can use it.",
];

// Converting adds a fresh personal organization and turns the converted one
// into a team organization. If the refreshed list can't be read, the converted
// entry is still updated so the switcher stops calling it personal; the
// provider picks the new personal organization up on its next load.
export async function getOrgsAfterConvert(
  converted: OrgResponse,
  currentOrgs: Org[],
): Promise<Org[]> {
  try {
    const response = await getV2ListUserOrganizations();
    if (response.status === 200) return response.data.map(normalizeOrg);
  } catch {
    // Fall through to patching the list we already have.
  }

  const convertedOrg = normalizeOrg(converted);
  if (!currentOrgs.some((org) => org.id === converted.id)) {
    return [...currentOrgs, convertedOrg];
  }
  return currentOrgs.map((org) =>
    org.id === converted.id ? convertedOrg : org,
  );
}
