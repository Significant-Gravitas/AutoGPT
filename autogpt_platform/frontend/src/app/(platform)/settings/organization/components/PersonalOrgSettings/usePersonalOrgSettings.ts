"use client";

import { useGetV2GetPersonalOrganizationContext } from "@/app/api/__generated__/endpoints/orgs/orgs";

// Accounts that only inherited organizations from someone else's team
// organization can open these settings but may not create or convert one.
export function usePersonalOrgSettings() {
  const { data: canCreateOrganization } =
    useGetV2GetPersonalOrganizationContext({
      query: {
        select: (res) =>
          res.status === 200 && res.data.can_create_organization === true,
      },
    });

  return { canCreateOrganization: canCreateOrganization ?? false };
}
