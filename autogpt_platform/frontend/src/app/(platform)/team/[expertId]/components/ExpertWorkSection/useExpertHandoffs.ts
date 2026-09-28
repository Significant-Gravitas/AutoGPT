import { useListDelegations } from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";

interface Args {
  expertId: string;
  enabled: boolean;
}

/** The hand-offs Otto (or a teammate) gave this expert, newest first. */
export function useExpertHandoffs({ expertId, enabled }: Args) {
  const query = useListDelegations(
    { expert_id: expertId },
    { query: { select: (res) => okData(res) ?? null, enabled } },
  );
  const delegations = query.data?.delegations ?? [];

  return {
    delegations,
    isWorkingForOtto: delegations.some(
      (d) => d.status === "running" && !d.delegated_by_expert_id,
    ),
    isLoading: query.isLoading,
    isError: query.isError && !query.data,
    refetch: query.refetch,
  };
}
