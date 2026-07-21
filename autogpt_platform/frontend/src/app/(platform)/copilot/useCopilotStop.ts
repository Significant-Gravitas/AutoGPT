import { postV2CancelSessionTask } from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import type { UseChatHelpers } from "@ai-sdk/react";
import type { UIMessage } from "ai";
import { hasActiveBackendStream, resolveInProgressTools } from "./helpers";

/**
 * User-visible marker appended to the last assistant message so the UI
 * renders a "You manually stopped this chat" row the moment the SSE is
 * aborted. The backend writes the same marker to the DB, so a subsequent
 * reload shows the same state. Must match `COPILOT_ERROR_PREFIX` in
 * `ChatMessagesContainer/helpers.ts`.
 */
const CANCELLED_MARKER = "[__COPILOT_ERROR_f7a1__] Operation cancelled";

interface UseCopilotStopArgs {
  sessionId: string | null;
  sdkStop: () => void;
  setMessages: UseChatHelpers<UIMessage>["setMessages"];
  /** Flipped to `true` so the stream's onError/onFinish callbacks don't
   *  misinterpret the resulting AbortError as a disconnect + reconnect. */
  isUserStoppingRef: React.MutableRefObject<boolean>;
  /** State setter mirroring ``isUserStoppingRef`` — drives the UI so the
   *  stop-button UX flips immediately instead of waiting for AI SDK's
   *  ``status`` to transition away from "streaming" on abort. */
  setIsUserStopping: (value: boolean) => void;
  /** True while the ``/cancel`` request is in flight. The flag-reset effect
   *  in ``useCopilotStream`` must not clear the user-stop flag from the
   *  stale pre-stop ``hasActiveStream`` cache while the backend is still
   *  draining the cancelled turn — doing so re-arms the auto-resume path,
   *  which replays the whole transcript over the locally-painted stopped
   *  state (duplicate messages + tool spinners flickering back). */
  isCancelInFlightRef: React.MutableRefObject<boolean>;
  /** Refetches the session after the cancel settles so ``hasActiveStream``
   *  reflects the confirmed stop instead of a stale cache. */
  refetchSession: () => Promise<{ data?: unknown }>;
}

/**
 * Build the `stop` handler for `useCopilotStream`.
 *
 * Wraps AI-SDK's `stop()` to:
 *   1. flag the stop as user-initiated (so onError/onFinish don't reconnect)
 *   2. abort the SSE fetch synchronously for instant UI feedback
 *   3. inject a cancellation marker into the visible assistant message
 *   4. asynchronously tell the backend executor to actually stop the task,
 *      surfacing a toast when the cancel was published but not yet
 *      confirmed (the task should stop shortly) or failed outright.
 */
export function useCopilotStop({
  sessionId,
  sdkStop,
  setMessages,
  isUserStoppingRef,
  setIsUserStopping,
  isCancelInFlightRef,
  refetchSession,
}: UseCopilotStopArgs) {
  async function stop() {
    isUserStoppingRef.current = true;
    setIsUserStopping(true);
    isCancelInFlightRef.current = true;
    try {
      try {
        sdkStop();
      } catch {
        // sdkStop throws if no fetch is in flight — the user-stop flag
        // already flipped, so the UI reflects the intent either way.
      }
      setMessages((prev) => {
        const resolved = resolveInProgressTools(prev, "cancelled");
        const last = resolved[resolved.length - 1];
        if (last?.role === "assistant") {
          return [
            ...resolved.slice(0, -1),
            {
              ...last,
              parts: [
                ...last.parts,
                { type: "text" as const, text: CANCELLED_MARKER },
              ],
            },
          ];
        }
        return resolved;
      });

      if (!sessionId) return;
      try {
        const res = await postV2CancelSessionTask(sessionId);
        if (
          res.status === 200 &&
          "reason" in res.data &&
          res.data.reason === "cancel_published_not_confirmed"
        ) {
          toast({
            title: "Stop may take a moment",
            description:
              "The cancel was sent but not yet confirmed. The task should stop shortly.",
          });
        }
      } catch {
        toast({
          title: "Could not stop the task",
          description: "The task may still be running in the background.",
          variant: "destructive",
        });
      }
    } finally {
      // Keep ``isCancelInFlightRef`` set through the confirming refetch. The
      // reset effect in useCopilotStream is gated on it, so clearing it before
      // the settled session state arrives would let that effect clear the
      // user-stop flag from the stale pre-stop cache and re-arm auto-resume
      // mid-drain. The cancel endpoint returns once the backend confirmed (or
      // force-completed) the stop, so this refetch reads the settled state.
      // Only clear the user-stop flag when the backend actually reports no
      // active stream.
      try {
        if (sessionId) {
          const result = await refetchSession();
          if (!hasActiveBackendStream(result)) {
            isUserStoppingRef.current = false;
            setIsUserStopping(false);
          }
        }
      } catch {
        // Leave the user-stop flag set; the fallback reset effect in
        // useCopilotStream clears it once a later session refetch reports
        // the stream gone.
      } finally {
        // Always release the guard once the refetch settles — leaving it set
        // would permanently disable the fallback reset effect.
        isCancelInFlightRef.current = false;
      }
    }
  }

  return stop;
}
