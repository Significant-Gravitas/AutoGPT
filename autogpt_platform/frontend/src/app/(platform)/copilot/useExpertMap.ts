import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useMemo } from "react";

export interface ExpertIdentity {
  name: string;
  avatarUrl: string | null;
  role: string | null;
}

export type ExpertIdentityMap = Map<string, ExpertIdentity>;

const EMPTY_MAP: ExpertIdentityMap = new Map();

export function useExpertMap() {
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const expertsQuery = useListExperts({
    query: {
      enabled: isExpertsEnabled,
      select: (response) =>
        response.status === 200 ? response.data : undefined,
    },
  });

  // Memoized on purpose: the identities read out of this map are passed as
  // props (`expertIdentity`) down the whole chat tree, so rebuilding it every
  // render would hand every consumer a fresh object identity each time.
  const expertsById = useMemo(() => {
    const experts = expertsQuery.data;
    if (!experts) return EMPTY_MAP;
    return new Map(
      experts.map((expert) => [
        expert.id,
        {
          name: expert.name,
          avatarUrl: expert.avatar_url ?? null,
          role: expert.role ?? null,
        },
      ]),
    );
  }, [expertsQuery.data]);

  return {
    expertsById: isExpertsEnabled ? expertsById : EMPTY_MAP,
    // A disabled query reports `isPending` forever, so both flags stay gated
    // on the feature flag to remain honest when experts are off.
    isLoadingExperts: isExpertsEnabled && expertsQuery.isPending,
    hasLoadedExperts: isExpertsEnabled && expertsQuery.isSuccess,
  };
}
