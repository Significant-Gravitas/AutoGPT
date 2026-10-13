import { useGetHomeDashboard } from "@/app/api/__generated__/endpoints/home/home";
import { okData } from "@/app/api/helpers";
import { useAttentionDecisions } from "@/app/(platform)/home/components/NeedsYou/useAttentionDecisions";
import { useHeldReview } from "@/app/(platform)/home/components/NeedsYou/useHeldReview";
import { useReviewCarousel } from "@/app/(platform)/home/components/NeedsYou/useReviewCarousel";

interface Args {
  expertId: string;
  enabled: boolean;
}

// Setup items are left to the Team page's Setup needed card, which names the fix.
export function useExpertNeedsYou({ expertId, enabled }: Args) {
  const { pendingIDs, decide } = useAttentionDecisions();
  const dashboardQuery = useGetHomeDashboard({
    query: {
      enabled,
      select: (res) =>
        (okData(res)?.attention ?? []).filter(
          (item) => item.expert?.id === expertId && item.kind !== "setup",
        ),
    },
  });
  const items = dashboardQuery.data ?? EMPTY;
  const held = useHeldReview({ items });
  const carousel = useReviewCarousel({ rows: held.rows, held });

  return {
    rows: held.rows,
    pendingCount: held.pendingCount,
    pendingIDs,
    decide,
    held,
    carousel,
  };
}

const EMPTY: never[] = [];
