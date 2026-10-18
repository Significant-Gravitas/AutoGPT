"use client";

import {
  getGetV2GetCopilotUsageQueryKey,
  getGetV2ListTurnsAtBudgetResetQueryKey,
  getV2ListTurnsAtBudgetReset,
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
  holdsMessage,
  isJoinedSendTooLong,
  returnToDraft,
  SCHEDULE_FAILED_TITLE,
  sendAtResetFailureTitle,
  SentNowError,
  sessionDraftKey,
} from "./helpers";
import { countSendChanges, startSendRead } from "./sendChanges";
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
  const scheduleSend = usePostV2ScheduleTurnAtBudgetReset({
    mutation: countSendChanges,
  });
  const cancelSend = useDeleteV2CancelTurnAtBudgetReset({
    mutation: countSendChanges,
  });
  const pending = jobs?.find((job) => job.kind === "send") ?? null;

  /** `onAnswered` hears as soon as the server answers, before a send it
   *  says to make now instead streams its reply. */
  async function schedule(message: string, onAnswered?: () => void) {
    if (!sessionId) throw new Error("This chat isn’t ready yet");
    // A chat sends one message at the reset; the server adds this one to
    // any already waiting and answers with the joined message.
    try {
      const response = await scheduleSend
        .mutateAsync({ sessionId, data: { kind: "send", message } })
        .finally(onAnswered);
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
  async function scheduleQuietly(message: string, onAnswered?: () => void) {
    try {
      await schedule(message, onAnswered);
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
      toastChangeFailure(error);
      return false;
    }
    await writeJobs((all) => all.filter((job) => job.kind !== "send"));
    return true;
  }

  async function edit() {
    const read = await readSend();
    // Without the server's copy, cancelling could drop text it holds.
    if ("error" in read) {
      toastChangeFailure(read.error);
      return;
    }
    // Nor with a copy that can predate a message this page added meanwhile.
    if (read.canHaveMissedAChange) {
      toastChangeFailure(new Error("It changed just now"));
      return;
    }
    const { send } = read;
    // Already sent, or cancelled elsewhere.
    if (!send) {
      rereadList();
      return;
    }
    if (!(await unschedule())) return;
    draft.write((current) => returnToDraft(current, send.message ?? ""));
    if (inputId) document.getElementById(inputId)?.focus();
  }

  /** The send as the server holds it now, since another tab or client may
   *  have added to it after this one last read the list. A request of its
   *  own: one the list's query is making is cancelled if the chat remounts,
   *  which would read as the server being unreachable. It stays out of the
   *  list's cache, where a change made while it was out is newer. */
  async function readSend() {
    if (!sessionId) return { send: pending, canHaveMissedAChange: false };
    const canHaveMissedAChange = startSendRead(sessionId);
    try {
      const response = await getV2ListTurnsAtBudgetReset(sessionId);
      if (response.status !== 200) return { error: response };
      return {
        send: response.data.jobs.find((job) => job.kind === "send") ?? null,
        canHaveMissedAChange: canHaveMissedAChange(),
      };
    } catch (error) {
      return { error };
    }
  }

  function rereadList() {
    void queryClient.invalidateQueries({
      queryKey: getGetV2ListTurnsAtBudgetResetQueryKey(sessionId ?? ""),
    });
  }

  function toastChangeFailure(error: unknown) {
    toast({
      title: "Couldn’t change your scheduled message",
      description: describeSendFailure(error, "it still sends at the reset"),
      variant: "destructive",
    });
  }

  /** For a message a page sent to be set and then lost track of, as on a
   *  reload mid-request: keep it if the server has it, otherwise put it back
   *  in the chat's draft. Never sent again from here, since the first try
   *  could still land and the server would add the message twice. */
  async function recover(message: string) {
    // A change this page makes meanwhile can't sway it: adding to the send
    // doesn't add this message, and Edit puts what it cancels in the draft.
    const read = await readSend();
    if ("send" in read && holdsMessage(read.send?.message ?? "", message)) {
      // The list, read as the chat opened, can predate the send landing.
      rereadList();
      return;
    }
    draft.write((current) => returnToDraft(current, message));
    const unread = "error" in read;
    toastSendFailure({
      title: unread
        ? "Couldn’t check your scheduled message"
        : SCHEDULE_FAILED_TITLE,
      // Sent again, a message the server may hold would be joined on twice.
      error: unread ? new Error("It may already be set to send") : undefined,
      recovery: "your message is back in the composer",
    });
  }

  return {
    pending,
    schedule,
    scheduleQuietly,
    recover,
    edit,
    cancel: unschedule,
    isBusy: scheduleSend.isPending || cancelSend.isPending,
  };
}
