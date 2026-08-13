import { useGetBriefingsGetLatestBriefing } from "@/app/api/__generated__/endpoints/briefings/briefings";
import { okData } from "@/app/api/helpers";

export function useCopilotHome() {
  const { data, isLoading, isError, refetch } =
    useGetBriefingsGetLatestBriefing({
      query: { select: (res) => okData(res) ?? null },
    });

  return {
    briefing: data ?? null,
    isLoadingBriefing: isLoading,
    isBriefingError: isError,
    refetchBriefing: refetch,
  };
}
