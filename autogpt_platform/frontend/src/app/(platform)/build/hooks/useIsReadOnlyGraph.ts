import { useGetV1GetSpecificGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";

// Read-only detection is a UX affordance only — the backend is the security
// boundary and already rejects mutations (save, etc.) from non-owners. This
// hook just hides controls that would otherwise fail silently.
export function useIsReadOnlyGraph() {
  const { user } = useSupabase();

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

  // Treat the loading state as not read-only to avoid flickering the banner
  // during initial mount; the canvas already shows a loading box at that stage.
  const isReadOnly = !!graph && !!user && graph.user_id !== user.id;

  return { isReadOnly };
}
