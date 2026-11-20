import { useListExpertCredentials } from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";

export function useExpertIntegrations(expertId: string | null) {
  const query = useListExpertCredentials(expertId ?? "", {
    query: {
      enabled: Boolean(expertId),
      select: (response) => okData(response) ?? [],
    },
  });

  return { integrations: query.data ?? [] };
}
