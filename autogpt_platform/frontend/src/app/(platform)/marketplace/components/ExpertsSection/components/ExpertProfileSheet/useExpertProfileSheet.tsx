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

  const isHired =
    expert !== null &&
    (!expert.is_template ||
      (expertsQuery.data ?? []).some(
        (hired) => hired.source_template_id === expert.id,
      ));

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

  return { isHired, isHiring, hire };
}
