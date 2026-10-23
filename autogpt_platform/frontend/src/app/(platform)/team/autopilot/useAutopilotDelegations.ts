import {
  useGetDelegationSettings,
  useListDelegations,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";
import { isToday } from "../components/DelegationList/helpers";

export function useAutopilotDelegations() {
  const listQuery = useListDelegations(undefined, {
    query: { select: (res) => okData(res) ?? null },
  });
  const settingsQuery = useGetDelegationSettings({
    query: { select: (res) => okData(res) ?? null },
  });

  const delegations = listQuery.data?.delegations ?? [];
  const todayCount = delegations.filter((d) => isToday(d.created_at)).length;

  return {
    delegations,
    summary: listQuery.data?.summary ?? null,
    todayCount,
    mode: settingsQuery.data?.mode ?? null,
    isLoading: listQuery.isLoading,
    isError: listQuery.isError && !listQuery.data,
    refetch: listQuery.refetch,
  };
}
