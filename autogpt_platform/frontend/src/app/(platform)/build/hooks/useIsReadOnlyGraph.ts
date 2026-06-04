import { useGetV1GetSpecificGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";

// Read-only detection is a UX affordance only — the backend is the security
// boundary and already rejects mutations (save, etc.) from non-owners. This
// hook just hides controls that would otherwise fail silently.
export function useIsReadOnlyGraph() {
  const { user, isUserLoading } = useSupabase();

  const [{ flowID, flowVersion }] = useQueryStates({
    flowID: parseAsString,
    flowVersion: parseAsInteger,
  });

  // Mirror useFlow's query (same key incl. version) so React Query serves both
  // from one cache entry and ownership reflects the version actually being
  // viewed, not just the latest.
  const { data: graph } = useGetV1GetSpecificGraph(
    flowID ?? "",
    flowVersion !== null ? { version: flowVersion } : {},
    {
      query: {
        select: (res) => res.data as GraphModel,
        enabled: !!flowID,
      },
    },
  );

  // Wait for both the graph and the auth state to resolve before deciding, so
  // the banner doesn't flicker for owners during initial mount. Once resolved,
  // anyone who isn't the confirmed owner (including a logged-out viewer) is
  // read-only — failing safe toward read-only rather than toward editable.
  const isReadOnly =
    !!graph && !isUserLoading && (!user || graph.user_id !== user.id);

  return { isReadOnly };
}
