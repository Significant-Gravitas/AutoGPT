import { useGetV2ListPresets } from "@/app/api/__generated__/endpoints/presets/presets";
import { okData } from "@/app/api/helpers";
import { retryUnlessClientError } from "../helpers";

/**
 * Single-page cap on the presets fetch. Beyond it, membership-based routing
 * degrades gracefully: `presetsComplete` turns false and unknown IDs fall
 * back to the by-ID preset fetch. Deemed acceptable — >100 presets on one
 * agent is an edge case; the escape hatch is unpaginating the endpoint.
 */
export const PRESETS_PAGE_SIZE = 100;

/**
 * The agent's presets (webhook triggers + templates). Shared by the sidebar
 * list and the detail-pane router so both read the same React Query cache
 * entry — keep all consumers on this hook so the query params can't drift.
 *
 * Returns the full page payload: `data.pagination.total_items` vs
 * `data.presets.length` tells callers whether the fetched page is the
 * complete set.
 */
export function useAgentPresetsQuery(graphId: string | undefined) {
  return useGetV2ListPresets(
    { graph_id: graphId ?? "", page: 1, page_size: PRESETS_PAGE_SIZE },
    {
      query: {
        enabled: !!graphId,
        select: okData,
        retry: retryUnlessClientError,
      },
    },
  );
}
