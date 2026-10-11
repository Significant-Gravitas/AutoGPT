"use client";

import {
  getGetV2GetCopilotUsageQueryKey,
  useDeleteV2CancelTurnAtBudgetReset,
  usePostV2ScheduleTurnAtBudgetReset,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { describeSendFailure } from "../ChatInput/helpers";
import { useChatDraft } from "../ChatInput/useChatDraft";
import {
  atResetRefusal,
  returnToDraft,
  SCHEDULE_FAILED_TITLE,
  sessionDraftKey,
} from "./helpers";
import { useAtResetJobs } from "./useAtResetJobs";

interface Args {
  sessionId: string | null;
  /** Sends right away instead, when the budget turns out to be back. */
  onSendNow?: (message: string) => void | Promise<void>;
  /** The composer to focus once Edit has put the message back in it. */
  inputId?: string;
}

export function useScheduledSend({ sessionId, onSendNow, inputId }: Args) {
  const queryClient = useQueryClient();
  const { jobs, writeJobs } = useAtResetJobs(sessionId);
  const [, setDraft] = useChatDraft(
    sessionId ? sessionDraftKey(sessionId) : undefined,
  );
  const scheduleSend = usePostV2ScheduleTurnAtBudgetReset();
  const cancelSend = useDeleteV2CancelTurnAtBudgetReset();

  async function schedule(message: string) {
    if (!sessionId) throw new Error("This chat isn’t ready yet");
    try {
      const response = await scheduleSend.mutateAsync({
        sessionId,
        data: { kind: "send", message },
      });
      if (response.status !== 200) return;
      const job = response.data;
      writeJobs((all) => [...all.filter((j) => j.kind !== "send"), job]);
    } catch (error) {
      const refusal = atResetRefusal(error);
      if (!refusal) throw error;
      void queryClient.invalidateQueries({
        queryKey: getGetV2GetCopilotUsageQueryKey(),
      });
      if (refusal === "not_out_of_budget" && onSendNow) {
        return onSendNow(message);
      }
      throw new Error(
        refusal === "no_reset"
          ? "Your budget doesn’t reset on its own"
          : "Your budget is already back",
      );
    }
  }

  /** For a message no composer is holding: a failure goes back into the
   *  chat's draft instead of being thrown. */
  async function scheduleQuietly(message: string) {
    try {
      await schedule(message);
    } catch (error) {
      setDraft((current) => returnToDraft(current, message));
      toast({
        title: SCHEDULE_FAILED_TITLE,
        description: describeSendFailure(
          error,
          "your message is back in the composer",
        ),
        variant: "destructive",
      });
    }
  }

  async function unschedule() {
    if (!sessionId) return false;
    try {
      await cancelSend.mutateAsync({ sessionId, kind: "send" });
    } catch (error) {
      toast({
        title: "Couldn’t change your scheduled message",
        description: describeSendFailure(error, "it still sends at the reset"),
        variant: "destructive",
      });
      return false;
    }
    writeJobs((all) => all.filter((job) => job.kind !== "send"));
    return true;
  }

  const pending = jobs?.find((job) => job.kind === "send") ?? null;

  async function edit() {
    const message = pending?.message ?? "";
    if (!(await unschedule())) return;
    setDraft((current) => returnToDraft(current, message));
    if (inputId) document.getElementById(inputId)?.focus();
  }

  return {
    pending,
    schedule,
    scheduleQuietly,
    edit,
    cancel: unschedule,
    isBusy: scheduleSend.isPending || cancelSend.isPending,
  };
}
