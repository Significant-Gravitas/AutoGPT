import { useGetV1GetSpecificGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { retryUnlessClientError } from "../helpers/graphLoadError";

// Read-only detection is a UX affordance only — the backend is the security
// boundary and already rejects mutations (save, etc.) from non-owners. This
// hook just hides controls that would otherwise fail silently.
export function useIsReadOnlyGraph() {
  const { user, isUserLoading } = useAuth();

  const [{ flowID, flowVersion }] = useQueryStates({
    flowID: parseAsString,
    flowVersion: parseAsInteger,
  });

  // Mirror useFlow's query (same key incl. version) so React Query serves both
  // from one cache entry and ownership reflects the version actually being
  // viewed, not just the latest.
  const { data: graph, isError: isGraphError } = useGetV1GetSpecificGraph(
    flowID ?? "",
    flowVersion !== null ? { version: flowVersion } : {},
    {
      query: {
        select: (res) => res.data as GraphModel,
        enabled: !!flowID,
        retry: retryUnlessClientError,
      },
    },
  );

  // Wait for both the graph and the auth state to resolve before deciding, so
  // the banner doesn't flicker for owners during initial mount. Once resolved,
  // anyone who isn't the confirmed owner (including a logged-out viewer) is
  // read-only — failing safe toward read-only rather than toward editable. A
  // fetch failure also fails safe to read-only, so a graph that couldn't load
  // never falls into the fully-editable "new agent" state.
  const isReadOnly =
    (!!flowID && isGraphError) ||
    (!!graph && !isUserLoading && (!user || graph.user_id !== user.id));

  return { isReadOnly };
}
