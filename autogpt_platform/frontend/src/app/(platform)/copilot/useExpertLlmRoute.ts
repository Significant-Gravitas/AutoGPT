import { useGetExpert } from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";
import { getExpertLlmRoute } from "./helpers/expertLlmRoute";

/** The pinned connection of the expert a new chat will address, if any. */
export function useExpertLlmRoute(expertId: string | null) {
  const query = useGetExpert(expertId ?? "", {
    query: {
      enabled: !!expertId,
      select: (res) => okData(res) ?? null,
    },
  });
  return {
    route: getExpertLlmRoute(query.data ?? null),
    isLoading: !!expertId && query.isLoading,
  };
}
