import {
  useGetDelegationSettings,
  useListDelegations,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";
import { isToday } from "../components/DelegationList/helpers";

interface Args {
  enabled: boolean;
}

export function useAutopilotDelegations({ enabled }: Args) {
  const listQuery = useListDelegations(undefined, {
    query: { select: (res) => okData(res) ?? null, enabled },
  });
  const settingsQuery = useGetDelegationSettings({
    query: { select: (res) => okData(res) ?? null, enabled },
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
