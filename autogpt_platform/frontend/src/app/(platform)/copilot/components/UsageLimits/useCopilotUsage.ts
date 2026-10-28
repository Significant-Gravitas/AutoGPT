import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";

export function useCopilotUsage() {
  return useGetV2GetCopilotUsage({
    query: {
      select: (response) =>
        response.status === 200 ? response.data : undefined,
      refetchInterval: 30_000,
      staleTime: 10_000,
    },
  });
}
