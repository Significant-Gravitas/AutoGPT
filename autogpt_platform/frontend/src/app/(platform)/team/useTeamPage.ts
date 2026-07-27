import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";

export function useTeamPage() {
  const expertsQuery = useListExperts({
    query: { select: (x) => x.data as Expert[] },
  });

  const hiredExperts = (expertsQuery.data ?? []).filter(
    (expert) => !expert.is_template && !expert.is_archived,
  );

  function installWorkflow(expertId: string) {
    void expertId;
  }

  return {
    hiredExperts,
    isLoading: expertsQuery.isLoading,
    isError: expertsQuery.isError,
    refetch: expertsQuery.refetch,
    installWorkflow,
  };
}
