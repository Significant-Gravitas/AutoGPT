import { useGetV2ListPresets } from "@/app/api/__generated__/endpoints/presets/presets";
import { okData } from "@/app/api/helpers";
import { retryUnlessClientError } from "../helpers";

// Page cap for agent presets query (addresses #13633)
export const PRESETS_PAGE_SIZE = 500;

/**
 * The agent's presets (webhook triggers + templates). Shared by the sidebar
 * and the detail-pane router so both read one React Query cache entry.
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
