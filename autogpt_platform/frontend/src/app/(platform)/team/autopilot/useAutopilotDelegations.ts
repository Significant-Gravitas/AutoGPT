import { useGetDelegationSettings } from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";
import { useDelegationPages } from "../components/DelegationList/useDelegationPages";

export function useAutopilotDelegations() {
  const pages = useDelegationPages();
  const settingsQuery = useGetDelegationSettings({
    query: { select: (res) => okData(res) ?? null },
  });

  return {
    ...pages,
    todayCount: pages.summary?.started_today ?? 0,
    mode: settingsQuery.data?.mode ?? null,
  };
}
