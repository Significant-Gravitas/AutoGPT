import { useGetV2GetSpecificBlocks } from "@/app/api/__generated__/endpoints/default/default";
import { okData } from "@/app/api/helpers";

export function useReviewBlockSchema(blockId: string | null | undefined) {
  const { data, isLoading } = useGetV2GetSpecificBlocks(
    { block_ids: blockId ? [blockId] : [] },
    { query: { enabled: !!blockId, select: okData, staleTime: Infinity } },
  );
  const schema = data?.[0]?.inputSchema as Record<string, unknown> | undefined;
  return { schema, isLoading: !!blockId && isLoading };
}
