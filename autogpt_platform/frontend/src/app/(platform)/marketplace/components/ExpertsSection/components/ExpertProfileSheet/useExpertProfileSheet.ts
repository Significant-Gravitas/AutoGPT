import {
  getListExpertsQueryKey,
  useHireExpert,
  useListExperts,
  useListExpertTemplates,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { HireResult } from "@/app/api/__generated__/models/hireResult";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { createElement } from "react";

export function useExpertProfileSheet(
  templateId: string | null,
  onClose: () => void,
) {
  const queryClient = useQueryClient();
  const router = useRouter();

  const templatesQuery = useListExpertTemplates({
    query: { select: (x) => x.data as Expert[] },
  });
  const expertsQuery = useListExperts({
    query: { select: (x) => x.data as Expert[] },
  });

  const template = (templatesQuery.data ?? []).find((t) => t.id === templateId);
  const isHired =
    templateId !== null &&
    (expertsQuery.data ?? []).some(
      (expert) => expert.source_template_id === templateId,
    );

  const { mutateAsync: hireExpert, isPending: isHiring } = useHireExpert();

  async function hire() {
    if (!templateId || !template) return;
    try {
      const response = await hireExpert({ data: { template_id: templateId } });
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
        action: createElement(
          "div",
          { className: "flex gap-2" },
          createElement(
            "button",
            {
              type: "button",
              className: "font-medium underline",
              onClick: () =>
                router.push(`/copilot?expertId=${result.expert.id}`),
            },
            `Chat with ${result.expert.name}`,
          ),
          createElement(
            "button",
            {
              type: "button",
              className: "font-medium underline",
              onClick: () => router.push("/team"),
            },
            "View team",
          ),
        ),
      });
      onClose();
    } catch {
      toast({
        title: `Couldn't hire ${template.name}`,
        description: "Something went wrong. Please try again.",
        variant: "destructive",
      });
    }
  }

  return { template, isHired, isHiring, hire };
}
