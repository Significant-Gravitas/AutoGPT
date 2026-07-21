"use client";

import {
  getGetV2GetSessionQueryKey,
  usePostV2CancelSessionTask,
} from "@/app/api/__generated__/endpoints/chat/chat";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { toast } from "@/components/molecules/Toast/use-toast";
import * as Sentry from "@sentry/nextjs";
import { HourglassIcon, XCircleIcon } from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";

interface Props {
  sessionID: string | null;
}

const QUEUED_TOOLTIP =
  "Will start automatically when one of your current tasks finishes.";

export function QueueBadge({ sessionID }: Props) {
  const queryClient = useQueryClient();
  const { mutate: cancelTask, isPending: isCancelling } =
    usePostV2CancelSessionTask({
      mutation: {
        onSuccess: () => {
          if (sessionID) {
            queryClient.invalidateQueries({
              queryKey: getGetV2GetSessionQueryKey(sessionID),
            });
          }
        },
        onError: (error) => {
          // 404 = session not found / not owned: sync the UI and
          // suppress the destructive toast.
          const status = (error as { response?: { status?: number } })?.response
            ?.status;
          if (status === 404) {
            if (sessionID) {
              queryClient.invalidateQueries({
                queryKey: getGetV2GetSessionQueryKey(sessionID),
              });
            }
            return;
          }
          Sentry.captureException(error);
          toast({
            variant: "destructive",
            title: "Could not cancel queued task",
            description: "Please try again.",
          });
        },
      },
    });

  function handleCancel() {
    if (!sessionID || isCancelling) return;
    cancelTask({ sessionId: sessionID });
  }

  return (
    <span className="inline-flex items-center gap-1">
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className="inline-flex items-center gap-1 rounded-full bg-purple-100 px-2 py-0.5 text-[11px] font-medium text-purple-800"
            data-testid="queue-badge-queued"
          >
            <HourglassIcon size={12} weight="bold" />
            Queued
          </span>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs whitespace-normal">
          {QUEUED_TOOLTIP}
        </TooltipContent>
      </Tooltip>
      {sessionID ? (
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              onClick={handleCancel}
              disabled={isCancelling}
              aria-label="Cancel queued task"
              data-testid="queue-cancel-button"
              className="inline-flex h-4 w-4 items-center justify-center rounded-full text-neutral-500 transition-colors hover:text-red-600 disabled:opacity-50"
            >
              <XCircleIcon size={14} weight="fill" />
            </button>
          </TooltipTrigger>
          <TooltipContent side="top">Cancel queued task</TooltipContent>
        </Tooltip>
      ) : null}
    </span>
  );
}
