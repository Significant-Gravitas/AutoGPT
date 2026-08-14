import {
  getListExpertsQueryKey,
  useHireExpert,
  useListExperts,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { HireResult } from "@/app/api/__generated__/models/hireResult";
import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";

export function useExpertProfileSheet(
  expert: Expert | null,
  onClose: () => void,
) {
  const queryClient = useQueryClient();

  const expertsQuery = useListExperts({
    query: { select: (x) => x.data as Expert[] },
  });

  // For templates the hired state is unknown until the experts query settles;
  // defaulting to [] would misreport an already-hired expert as hireable.
  const hiredLookup: "loading" | "error" | "loaded" =
    expert === null || !expert.is_template
      ? "loaded"
      : expertsQuery.isError
        ? "error"
        : expertsQuery.isSuccess
          ? "loaded"
          : "loading";

  const hiredExpert =
    expert === null
      ? null
      : expert.is_template
        ? ((expertsQuery.data ?? []).find(
            (hired) => hired.source_template_id === expert.id,
          ) ?? null)
        : expert;

  const isHired = hiredLookup === "loaded" && hiredExpert !== null;

  const { mutateAsync: hireExpert, isPending: isHiring } = useHireExpert();

  async function hire() {
    if (!expert || !expert.is_template) return;
    try {
      const response = await hireExpert({ data: { template_id: expert.id } });
      const result = response.data as HireResult;
      await queryClient.invalidateQueries({
        queryKey: getListExpertsQueryKey(),
      });
      toast({
        title: `${result.expert.name} joined your team`,
        description: result.failed_preloads.length
          ? `Couldn't attach: ${result.failed_preloads.join(", ")}`
          : undefined,
        variant: "success",
        action: (
          <div className="flex gap-2">
            <Button
              as="NextLink"
              href={`/copilot?expertId=${result.expert.id}`}
              variant="secondary"
              size="small"
              unmask={false}
            >
              {`Chat with ${result.expert.name}`}
            </Button>
            <Button as="NextLink" href="/team" variant="ghost" size="small">
              View team
            </Button>
          </div>
        ),
      });
      onClose();
    } catch {
      toast({
        title: `Couldn't hire ${expert.name}`,
        description: "Something went wrong. Please try again.",
        variant: "destructive",
      });
    }
  }

  function retryHiredLookup() {
    void expertsQuery.refetch();
  }

  return {
    isHired,
    isHiring,
    hire,
    hiredExpertId: hiredExpert?.id ?? null,
    hiredLookup,
    retryHiredLookup,
  };
}
