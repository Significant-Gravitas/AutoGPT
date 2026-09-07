import { useGetHomeDashboard } from "@/app/api/__generated__/endpoints/home/home";
import { okData } from "@/app/api/helpers";
import { useAttentionDecisions } from "@/app/(platform)/home/components/NeedsYou/useAttentionDecisions";

interface Args {
  expertId: string;
  enabled: boolean;
}

export function useExpertNeedsYou({ expertId, enabled }: Args) {
  const { pendingIDs, decide } = useAttentionDecisions();
  const dashboardQuery = useGetHomeDashboard({
    query: {
      enabled,
      select: (res) =>
        (okData(res)?.attention ?? []).filter(
          (item) => item.expert?.id === expertId,
        ),
    },
  });

  return {
    items: dashboardQuery.data ?? [],
    pendingIDs,
    decide,
  };
}
