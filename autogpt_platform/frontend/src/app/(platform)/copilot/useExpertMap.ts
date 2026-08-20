import { useListExpertIdentities } from "@/app/api/__generated__/endpoints/experts/experts";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useMemo } from "react";

/**
 * Why a thread is read-only. `fired` is the only reason we can state as fact;
 * `unavailable` is a transient roster-read failure and `unknown` covers an id
 * that is simply absent from a settled roster (deleted, wrong, or never ours).
 */
export type ExpertReadOnlyReason = "fired" | "unavailable" | "unknown";

export interface ExpertIdentity {
  id: string;
  name: string;
  avatarUrl: string | null;
  role: string | null;
  isArchived: boolean;
  readOnlyReason: ExpertReadOnlyReason | null;
}

export type ExpertIdentityMap = Map<string, ExpertIdentity>;

const EMPTY_MAP: ExpertIdentityMap = new Map();
const EMPTY_EXPERTS: ExpertIdentity[] = [];
const EMPTY_IDS: ReadonlySet<string> = new Set();

const FALLBACK_ARCHIVED_NAME = "This expert";

/**
 * Resolve the identity a chat session should render for `activeExpertId`.
 *
 * Fail closed: once the roster query has SETTLED (success or error), a
 * session pointing at an expert we can't resolve is treated as archived
 * (read-only history, generic name) — never a writable plain Autopilot
 * thread. A cached identity can still supply its name after a failed
 * refetch, but is marked unavailable and read-only. Passing the settled
 * flag rather than the success flag is load-bearing: a failed roster fetch
 * must not fail open into a writable composer.
 */
export function resolveExpertIdentity(
  activeExpertId: string | null,
  expertsById: ExpertIdentityMap,
  { settled, errored }: { settled: boolean; errored: boolean },
): ExpertIdentity | null {
  if (!activeExpertId) return null;
  const found = expertsById.get(activeExpertId);
  if (found) {
    return errored
      ? { ...found, isArchived: true, readOnlyReason: "unavailable" }
      : found;
  }
  if (!settled) return null;
  return {
    id: activeExpertId,
    name: FALLBACK_ARCHIVED_NAME,
    avatarUrl: null,
    role: null,
    isArchived: true,
    readOnlyReason: errored ? "unavailable" : "unknown",
  };
}

/**
 * Identity map for expert-scoped chat sessions.
 *
 * CONTRACT: `expertsById` deliberately includes ARCHIVED experts — it is the
 * identity source for read-only fired-expert threads, so it must keep
 * resolving names after an expert leaves the active roster or a background
 * refetch fails. Consumers that present experts as addressable receive the
 * separate `activeExperts` projection instead of iterating this map.
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
  const expertCollections = useMemo(() => {
    const experts = expertsQuery.data;
    if (!experts)
      return {
        expertsById: EMPTY_MAP,
        activeExperts: EMPTY_EXPERTS,
        activeExpertIds: EMPTY_IDS,
      };
    const identities = experts.map((expert) => ({
      id: expert.id,
      name: expert.name,
      avatarUrl: expert.avatar_url ?? null,
      role: expert.role,
      isArchived: expert.is_archived,
      readOnlyReason: expert.is_archived ? ("fired" as const) : null,
    }));
    const activeExperts = identities.filter((expert) => !expert.isArchived);
    return {
      expertsById: new Map(identities.map((expert) => [expert.id, expert])),
      activeExperts,
      activeExpertIds: new Set(activeExperts.map((expert) => expert.id)),
    };
  }, [expertsQuery.data]);

  const canAddressExperts = isExpertsEnabled && !expertsQuery.isError;

  return {
    expertsById: isExpertsEnabled ? expertCollections.expertsById : EMPTY_MAP,
    activeExperts: canAddressExperts
      ? expertCollections.activeExperts
      : EMPTY_EXPERTS,
    // Same membership as `activeExperts`, as a set, so callers testing "can the
    // user still address this id?" do a constant-time lookup instead of a scan.
    activeExpertIds: canAddressExperts
      ? expertCollections.activeExpertIds
      : EMPTY_IDS,
    // A disabled query reports `isPending` forever, so all flags stay gated
    // on the feature flag to remain honest when experts are off.
    isLoadingExperts: isExpertsEnabled && expertsQuery.isFetching,
    hasExpertsErrored:
      isExpertsEnabled && expertsQuery.isError && !expertsQuery.isFetching,
    // Settled = success OR error. resolveExpertIdentity keys off this so an
    // errored roster fetch fails closed (read-only) instead of open.
    hasExpertsSettled:
      isExpertsEnabled &&
      !expertsQuery.isFetching &&
      (expertsQuery.isSuccess || expertsQuery.isError),
  };
}
