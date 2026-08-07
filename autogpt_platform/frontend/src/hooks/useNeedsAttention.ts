import { useGetV2GetPendingReviews } from "@/app/api/__generated__/endpoints/executions/executions";
import { okData } from "@/app/api/helpers";

export function useNeedsAttention() {
  const { data, isLoading } = useGetV2GetPendingReviews(
    { page: 1, page_size: 100 },
    { query: { select: (res) => okData(res) ?? [], refetchInterval: 30_000 } },
  );
  const reviews = data ?? [];
  return { reviews, count: reviews.length, isLoading };
}
