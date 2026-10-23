import { useListDelegations } from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";

/** The hand-offs Otto (or a teammate) gave this expert, newest first. */
export function useExpertHandoffs(expertId: string) {
  const query = useListDelegations(
    { expert_id: expertId },
    { query: { select: (res) => okData(res) ?? null } },
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
