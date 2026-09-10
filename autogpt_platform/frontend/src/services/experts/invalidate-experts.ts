import {
  getListExpertIdentitiesQueryKey,
  getListExpertsQueryKey,
} from "@/app/api/__generated__/endpoints/experts/experts";
import type { QueryClient } from "@tanstack/react-query";

export function invalidateExpertRosterQueries(queryClient: QueryClient) {
  return Promise.all([
    queryClient.invalidateQueries({ queryKey: getListExpertsQueryKey() }),
    queryClient.invalidateQueries({
      queryKey: getListExpertIdentitiesQueryKey(),
    }),
  ]);
}
