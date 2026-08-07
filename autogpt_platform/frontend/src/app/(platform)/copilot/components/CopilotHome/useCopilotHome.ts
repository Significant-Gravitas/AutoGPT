import { useGetBriefingsGetLatestBriefing } from "@/app/api/__generated__/endpoints/briefings/briefings";
import { okData } from "@/app/api/helpers";

export function useCopilotHome() {
  const { data, isLoading } = useGetBriefingsGetLatestBriefing({
    query: { select: (res) => okData(res) ?? null },
  });

  return {
    briefing: data ?? null,
    isLoadingBriefing: isLoading,
    hasBriefing: Boolean(data),
  };
}
