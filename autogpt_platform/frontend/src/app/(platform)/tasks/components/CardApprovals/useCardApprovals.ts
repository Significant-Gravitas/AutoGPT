import type { CardApproval } from "@/app/api/__generated__/models/cardApproval";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useAnswerApprovals } from "../../data/useTasksData";

export function useCardApprovals(taskId: string, cards: CardApproval[]) {
  const answer = useAnswerApprovals();

  function decideAll(approved: boolean) {
    if (answer.isPending) return;
    answer.mutate(
      {
        id: taskId,
        decisions: cards.map((card) => ({
          confirmation_id: card.confirmation_id,
          approved,
        })),
      },
      {
        onError: (error) =>
          toast({
            title: approved
              ? "Couldn't approve the cards"
              : "Couldn't decline the cards",
            description: error instanceof Error ? error.message : undefined,
            variant: "destructive",
          }),
      },
    );
  }

  return {
    approve: () => decideAll(true),
    decline: () => decideAll(false),
    pending: answer.isPending
      ? answer.variables?.decisions[0]?.approved
        ? "approve"
        : "decline"
      : null,
  };
}
