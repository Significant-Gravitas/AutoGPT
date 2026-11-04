"use client";

import { useGetAdminListAnAccountSTeamOrganizations } from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";
import { useQueryState } from "nuqs";
import { ACCOUNT_QUERY_PARAM } from "../../helpers";
import { getErrorStatus, shouldRetryLookup } from "./helpers";

export function useTeamPlans() {
  const [account, setAccount] = useQueryState(ACCOUNT_QUERY_PARAM);
  const email = account?.trim() || null;
  const orgsQuery = useGetAdminListAnAccountSTeamOrganizations(
    email ? { email } : undefined,
    {
      query: {
        select: okData,
        enabled: email !== null,
        retry: shouldRetryLookup,
      },
    },
  );

  function lookUp(value: string) {
    const next = value.trim() || null;
    if (next !== null && next === email) {
      void orgsQuery.refetch();
      return;
    }
    void setAccount(next);
  }

  return {
    email,
    orgs: orgsQuery.data,
    isLoading: orgsQuery.isLoading,
    isNotFound: getErrorStatus(orgsQuery.error) === 404,
    error: orgsQuery.error,
    retry: () => void orgsQuery.refetch(),
    lookUp,
  };
}
