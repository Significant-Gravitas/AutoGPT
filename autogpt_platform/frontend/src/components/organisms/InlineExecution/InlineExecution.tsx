"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { useCopilotChatActions } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/useCopilotChatActions";
import { ExecutionOutputs } from "./ExecutionOutputs";
import { useInlineExecution } from "./useInlineExecution";

interface Props {
  graphId: string;
  executionId: string;
  title: string;
  disabled?: boolean;
}
export function InlineExecution({
  graphId,
  executionId,
  title,
  disabled = false,
}: Props) {
  const { chatSurface, onSend } = useCopilotChatActions();
  const shared = chatSurface === "share";
  const run = useInlineExecution(graphId, executionId, !shared);
  const [sending, setSending] = useState(false);
  const [sendError, setSendError] = useState("");
  async function retry() {
    if (sending || disabled || shared) return;
    setSending(true);
    setSendError("");
    try {
      await onSend(
        `Help me retry failed execution ${executionId} of graph ${graphId}. Review the failure and any required approvals before running it again.`,
      );
    } catch {
      setSendError("The retry request could not be sent. Try again.");
    } finally {
      setSending(false);
    }
  }
  return (
    <section aria-label={title} className="min-w-0 space-y-3">
      <h3 className="font-semibold">{title}</h3>
      {shared ? (
        <p className="text-sm">
          Live execution details are available in the original conversation.
        </p>
      ) : (
        <>
          <p role="status" className="text-sm">
            {run.query.isPending
              ? "Loading execution…"
              : run.query.isError || !run.execution
                ? "Execution details are unavailable. Access may have changed."
                : `Status: ${run.execution.status.toLowerCase()}`}
          </p>
          {run.execution?.ended_at && (
            <p className="text-xs text-muted-foreground">
              Finished {new Date(run.execution.ended_at).toLocaleString()}
            </p>
          )}
          {run.error && (
            <p role="alert" className="text-sm text-destructive">
              {run.error}
            </p>
          )}
          {sendError && (
            <p role="alert" className="text-sm text-destructive">
              {sendError}
            </p>
          )}
          {run.cancelRequested && run.running && (
            <p role="status" className="text-sm">
              Stop requested; waiting for the backend to confirm.
            </p>
          )}
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              variant="secondary"
              size="small"
              onClick={() => void run.query.refetch()}
              disabled={run.query.isFetching}
            >
              Refresh status
            </Button>
            {run.running && (
              <Button
                type="button"
                variant="secondary"
                size="small"
                onClick={() => void run.cancel()}
                disabled={disabled || run.cancelling || run.cancelRequested}
              >
                Stop execution
              </Button>
            )}
            {run.execution &&
              [
                "FAILED",
                "TERMINATED",
                "CANCELLED",
                "STOPPED",
                "TIMED_OUT",
              ].includes(run.execution.status) && (
                <Button
                  type="button"
                  size="small"
                  onClick={() => void retry()}
                  disabled={disabled || sending}
                >
                  Review and retry in chat
                </Button>
              )}
          </div>
          {run.execution && (
            <ExecutionOutputs outputs={run.execution.outputs} />
          )}
        </>
      )}
    </section>
  );
}
