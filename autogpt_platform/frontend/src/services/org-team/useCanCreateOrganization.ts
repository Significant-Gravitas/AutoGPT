"use client";

import { useGetV2GetPersonalOrganizationContext } from "@/app/api/__generated__/endpoints/orgs/orgs";

interface Args {
  enabled?: boolean;
}

// Whether the account may create or convert an organization: only with
// access of its own, not access inherited through someone else's team
// organization. Anything but a yes from the backend reads as no.
export function useCanCreateOrganization({ enabled = true }: Args = {}) {
  const { data } = useGetV2GetPersonalOrganizationContext({
    query: {
      enabled,
      select: (res) =>
        res.status === 200 && res.data.can_create_organization === true,
    },
  });

  return data === true;
}
