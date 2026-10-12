import {
  getGetExpertQueryKey,
  getListExpertCredentialsQueryKey,
  getListExpertIdentitiesQueryKey,
  getListExpertSetupItemsQueryKey,
  getListExpertsQueryKey,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { invalidateAllScheduleQueries } from "@/services/schedules/invalidate-schedules";
import type { QueryClient } from "@tanstack/react-query";

export function invalidateExpertRosterQueries(queryClient: QueryClient) {
  return Promise.all([
    queryClient.invalidateQueries({ queryKey: getListExpertsQueryKey() }),
    queryClient.invalidateQueries({
      queryKey: getListExpertIdentitiesQueryKey(),
    }),
  ]);
}

// Granting or revoking a credential changes more than the expert's own
// integrations list: the backend creates the schedules the missing grant was
// blocking, which the expert's workflows, the roster's schedule counts, the
// schedule lists and the Team page's setup card all read. Every grant site
// must invalidate all of them, or connecting on one page leaves the other
// page's list stale until a reload.
export function invalidateExpertGrantQueries(
  queryClient: QueryClient,
  expertId: string,
) {
  return Promise.all([
    queryClient.invalidateQueries({
      queryKey: getListExpertCredentialsQueryKey(expertId),
    }),
    queryClient.invalidateQueries({ queryKey: getGetExpertQueryKey(expertId) }),
    queryClient.invalidateQueries({
      queryKey: getListExpertSetupItemsQueryKey(),
    }),
    invalidateExpertRosterQueries(queryClient),
    invalidateAllScheduleQueries(queryClient),
  ]);
}
