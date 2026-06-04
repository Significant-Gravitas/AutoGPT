import { useGetV1GetSpecificGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { parseAsString, useQueryStates } from "nuqs";

export function useIsReadOnlyGraph() {
  const { user } = useSupabase();

  const [{ flowID }] = useQueryStates({
    flowID: parseAsString,
  });

  const { data: graph } = useGetV1GetSpecificGraph(
    flowID ?? "",
    {},
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
