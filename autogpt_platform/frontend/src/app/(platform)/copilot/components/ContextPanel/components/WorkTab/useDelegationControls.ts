"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useContext, useState } from "react";
import {
  getGetV2GetSessionQueryKey,
  postV2CancelSessionTask,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { CopilotChatActionsContext } from "../../../CopilotChatActionsProvider/useCopilotChatActions";
import { retryMessage } from "../../../DelegationStatusLine/helpers";

/** What the panel can do about a hand-off: stop it, or ask Otto to. */
export function useDelegationControls(
  subSessionId: string | null,
  name: string,
  chatSessionId: string | null,
) {
  const queryClient = useQueryClient();
  const actions = useContext(CopilotChatActionsContext);
  const [isCancelling, setIsCancelling] = useState(false);

  function askOtto(message: string) {
    if (!actions) return;
    void Promise.resolve(actions.onSend(message)).catch(() =>
      toast({ title: "Couldn't send message", variant: "destructive" }),
    );
  }

  async function cancel() {
    if (!subSessionId || isCancelling) return;
    setIsCancelling(true);
    try {
      const res = await postV2CancelSessionTask(subSessionId);
      if (res.status !== 200) throw new Error("cancel failed");
    } catch {
      toast({
        title: `Couldn't stop ${name}`,
        description: "They may keep working in their own thread.",
        variant: "destructive",
      });
    } finally {
      await Promise.all(
        [subSessionId, chatSessionId]
          .filter((id): id is string => !!id)
          .map((id) =>
            queryClient.invalidateQueries({
              queryKey: getGetV2GetSessionQueryKey(id),
            }),
          ),
      );
      setIsCancelling(false);
    }
  }

  return {
    canAskOtto: !!actions,
    isCancelling,
    cancel,
    nudge: () => askOtto(`Please check on ${name}'s hand-off.`),
    redelegate: () => askOtto(`Please re-delegate the same brief to ${name}.`),
    retry: () => askOtto(retryMessage(name)),
    raiseBudget: () =>
      askOtto(`Raise the cap and retry the hand-off to ${name}.`),
  };
}
