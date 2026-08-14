import { useListExpertIdentities } from "@/app/api/__generated__/endpoints/experts/experts";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useMemo } from "react";

export interface ExpertIdentity {
  id: string;
  name: string;
  avatarUrl: string | null;
  role: string | null;
  isArchived: boolean;
}

export type ExpertIdentityMap = Map<string, ExpertIdentity>;

const EMPTY_MAP: ExpertIdentityMap = new Map();

const FALLBACK_ARCHIVED_NAME = "This expert";

/**
 * Resolve the identity a chat session should render for `activeExpertId`.
 *
 * Fail closed: once the roster query has SETTLED (success or error), a
 * session pointing at an expert we can't resolve is treated as archived
 * (read-only history, generic name) — never a writable plain Autopilot
 * thread. Passing the settled flag rather than the success flag is
 * load-bearing: a failed roster fetch must not fail open into a writable
 * composer.
 */
export function resolveExpertIdentity(
  activeExpertId: string | null,
  expertsById: ExpertIdentityMap,
  hasExpertsSettled: boolean,
): ExpertIdentity | null {
  if (!activeExpertId) return null;
  const found = expertsById.get(activeExpertId);
  if (found) return found;
  if (!hasExpertsSettled) return null;
  return {
    id: activeExpertId,
    name: FALLBACK_ARCHIVED_NAME,
    avatarUrl: null,
    role: null,
    isArchived: true,
  };
}

/**
 * Active-roster projection of the identity map. `expertsById` includes
 * ARCHIVED experts by contract (see useExpertMap) — any consumer that lets
 * the user address or pick an expert must go through this filter instead of
 * iterating the raw map.
 */
export function getActiveExperts(
  expertsById: ExpertIdentityMap,
): ExpertIdentity[] {
  return [...expertsById.values()].filter((expert) => !expert.isArchived);
}

/**
 * Identity map for expert-scoped chat sessions.
 *
 * CONTRACT: `expertsById` deliberately includes ARCHIVED experts — it is the
 * identity source for read-only fired-expert threads, so it must keep
 * resolving names after an expert leaves the active roster. Consumers that
 * present experts as addressable (pickers, rosters, mention lists) must use
 * `getActiveExperts` instead of iterating the map directly.
 */
export function useExpertMap() {
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const expertsQuery = useListExpertIdentities({
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
          id: expert.id,
          name: expert.name,
          avatarUrl: expert.avatar_url ?? null,
          role: expert.role ?? null,
          isArchived: expert.is_archived,
        },
      ]),
    );
  }, [expertsQuery.data]);

  return {
    expertsById:
      isExpertsEnabled && !expertsQuery.isError ? expertsById : EMPTY_MAP,
    // A disabled query reports `isPending` forever, so all flags stay gated
    // on the feature flag to remain honest when experts are off.
    isLoadingExperts: isExpertsEnabled && expertsQuery.isFetching,
    hasLoadedExperts:
      isExpertsEnabled && expertsQuery.isSuccess && !expertsQuery.isFetching,
    // Settled = success OR error. resolveExpertIdentity keys off this so an
    // errored roster fetch fails closed (read-only) instead of open.
    hasExpertsSettled:
      isExpertsEnabled &&
      !expertsQuery.isFetching &&
      (expertsQuery.isSuccess || expertsQuery.isError),
  };
}
