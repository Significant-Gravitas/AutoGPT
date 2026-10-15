"use client";

import {
  getGetV2GetCopilotUsageQueryKey,
  useDeleteV2CancelTurnAtBudgetReset,
  usePostV2ScheduleTurnAtBudgetReset,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { toastSendFailure } from "../../helpers/sendFailureToast";
import { describeSendFailure } from "../ChatInput/helpers";
import { useChatDraftAccess } from "../ChatInput/useChatDraft";
import {
  atResetRefusal,
  isJoinedSendTooLong,
  returnToDraft,
  sendAtResetFailureTitle,
  SentNowError,
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
  const draft = useChatDraftAccess(
    sessionId ? sessionDraftKey(sessionId) : undefined,
  );
  const scheduleSend = usePostV2ScheduleTurnAtBudgetReset();
  const cancelSend = useDeleteV2CancelTurnAtBudgetReset();
  const pending = jobs?.find((job) => job.kind === "send") ?? null;

  async function schedule(message: string) {
    if (!sessionId) throw new Error("This chat isn’t ready yet");
    // A chat sends one message at the reset; the server adds this one to
    // any already waiting and answers with the joined message.
    try {
      const response = await scheduleSend.mutateAsync({
        sessionId,
        data: { kind: "send", message },
      });
      if (response.status !== 200) return;
      const job = response.data;
      await writeJobs((all) => [...all.filter((j) => j.kind !== "send"), job]);
    } catch (error) {
      if (isJoinedSendTooLong(error)) {
        throw new Error("Too long to add to the message already waiting");
      }
      const refusal = atResetRefusal(error);
      if (!refusal) throw error;
      void queryClient.invalidateQueries({
        queryKey: getGetV2GetCopilotUsageQueryKey(),
      });
      if (refusal === "not_out_of_budget" && onSendNow) {
        try {
          return await onSendNow(message);
        } catch (sendError) {
          throw new SentNowError(sendError);
        }
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
      draft.write((current) => returnToDraft(current, message));
      toastSendFailure({
        title: sendAtResetFailureTitle(error),
        error,
        recovery: "your message is back in the composer",
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
    await writeJobs((all) => all.filter((job) => job.kind !== "send"));
    return true;
  }

  async function edit() {
    const message = pending?.message ?? "";
    if (!(await unschedule())) return;
    draft.write((current) => returnToDraft(current, message));
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
