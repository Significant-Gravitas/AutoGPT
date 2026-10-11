"use client";

import { useQueryClient } from "@tanstack/react-query";

import {
  getGetV2GetCopilotUsageQueryKey,
  usePostV2ContinueSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";

import { useCopilotUIStore } from "../../../../store";
import { continueOutcome } from "./helpers";

interface Args {
  sessionId: string;
  /** Attach to the turn the server started. */
  onFollow: () => void;
  /** The account ran out of budget since the pause; when it next resets. */
  onOutOfBudget: (resetsAt: string | null) => void;
}

export function useKeepGoing({ sessionId, onFollow, onOutOfBudget }: Args) {
  const queryClient = useQueryClient();
  // The composer's model choice, as a message sent now would carry it.
  const model = useCopilotUIStore((state) => state.copilotLlmModel);
  const { mutateAsync, isPending } = usePostV2ContinueSession();

  async function keepGoing() {
    try {
      await mutateAsync({ sessionId, data: { model } });
      onFollow();
    } catch (error) {
      const outcome = continueOutcome(error);
      if (outcome.kind === "running") {
        onFollow();
        return;
      }
      if (outcome.kind === "out_of_budget") {
        void queryClient.invalidateQueries({
          queryKey: getGetV2GetCopilotUsageQueryKey(),
        });
        onOutOfBudget(outcome.resetsAt);
        return;
      }
      toast({
        title: "Couldn't pick this back up",
        description: outcome.description,
        variant: "destructive",
      });
    }
  }

  return { keepGoing, isContinuing: isPending };
}
