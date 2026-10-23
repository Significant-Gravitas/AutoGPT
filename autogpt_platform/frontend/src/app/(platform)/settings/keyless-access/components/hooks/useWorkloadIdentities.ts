"use client";

import {
  getGetV1ListWorkloadIdentitiesQueryKey,
  useGetV1ListWorkloadIdentities,
} from "@/app/api/__generated__/endpoints/workload-identity/workload-identity";
import { WorkloadIdentityStatus } from "@/app/api/__generated__/models/workloadIdentityStatus";

export const WORKLOAD_IDENTITIES_QUERY_KEY =
  getGetV1ListWorkloadIdentitiesQueryKey();

export function useWorkloadIdentities() {
  const query = useGetV1ListWorkloadIdentities({
    query: {
      select: (response) =>
        response.status === 200
          ? response.data.filter(
              (identity) => identity.status !== WorkloadIdentityStatus.REVOKED,
            )
          : [],
    },
  });

  const identities = query.data ?? [];

  return {
    identities,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
    isEmpty: !query.isLoading && !query.isError && identities.length === 0,
  };
}
