import { useUninstallExpertWorkflow } from "@/app/api/__generated__/endpoints/experts/experts";
import { toast } from "@/components/molecules/Toast/use-toast";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";
import { invalidateAllScheduleQueries } from "@/services/schedules/invalidate-schedules";
import { useQueryClient } from "@tanstack/react-query";

interface Args {
  expertId: string;
  expertName: string;
  workflowId: string;
  workflowName: string;
  graphID: string | null;
  onClose: () => void;
}

export function useRemoveWorkflowDialog({
  expertId,
  expertName,
  workflowId,
  workflowName,
  graphID,
  onClose,
}: Args) {
  const queryClient = useQueryClient();

  const { mutate, isPending: isRemoving } = useUninstallExpertWorkflow({
    mutation: {
      onSuccess: async () => {
        await Promise.all([
          invalidateExpertRosterQueries(queryClient),
          invalidateAllScheduleQueries(queryClient, graphID ?? undefined),
        ]);
        toast({
          title: `${workflowName} removed from ${expertName}`,
          description: "The agent is still in your library.",
        });
        onClose();
      },
      onError: () => {
        toast({
          title: `Could not remove ${workflowName}`,
          description: `${expertName} is still running it. Please try again.`,
          variant: "destructive",
        });
      },
    },
  });

  function handleRemove() {
    mutate({ expertId, workflowId });
  }

  return { isRemoving, handleRemove };
}
