import { getGetHomeDashboardQueryKey } from "@/app/api/__generated__/endpoints/home/home";
import {
  getListPausedScheduledChatsQueryKey,
  useDeleteV1DeleteExecutionSchedule,
  useResumeScheduledChat,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { PausedScheduledChat } from "@/app/api/__generated__/models/pausedScheduledChat";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { invalidateAllScheduleQueries } from "@/services/schedules/invalidate-schedules";
import * as Sentry from "@sentry/nextjs";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

interface Args {
  paused: PausedScheduledChat;
}

export function usePausedChatListItem({ paused }: Args) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);

  function refresh() {
    queryClient.invalidateQueries({
      queryKey: getListPausedScheduledChatsQueryKey(),
    });
    // Home lists each paused chat under Needs you.
    queryClient.invalidateQueries({ queryKey: getGetHomeDashboardQueryKey() });
    return invalidateAllScheduleQueries(queryClient);
  }

  const { mutate, isPending } = useResumeScheduledChat({
    mutation: {
      onSuccess: () => {
        void refresh();
        toast({
          title: "Resumed",
          description: `“${paused.title}” runs again at its next scheduled time.`,
        });
      },
      onError: (error) => {
        void refresh();
        // Resumed elsewhere already: the refresh drops the row.
        if (error instanceof ApiError && error.status === 404) return;
        Sentry.captureException(error);
        toast({
          variant: "destructive",
          title: "Couldn't resume this scheduled chat",
          description: "Please try again.",
        });
      },
    },
  });

  function handleDeleted() {
    setIsDeleteOpen(false);
    void refresh();
    toast({ title: "Scheduled chat deleted" });
  }

  const { mutate: deleteSchedule, isPending: isDeleting } =
    useDeleteV1DeleteExecutionSchedule({
      mutation: {
        onSuccess: handleDeleted,
        onError: (error) => {
          // Its schedule was gone already; the delete still cleared the row.
          if (error instanceof ApiError && error.status === 404) {
            handleDeleted();
            return;
          }
          void refresh();
          Sentry.captureException(error);
          toast({
            variant: "destructive",
            title: "Couldn't delete this scheduled chat",
            description: "Please try again.",
          });
        },
      },
    });

  const sessionHref = paused.session_id
    ? `/copilot?sessionId=${encodeURIComponent(paused.session_id)}`
    : null;

  function handleResume() {
    if (isPending) return;
    mutate({ scheduleKey: paused.schedule_key });
  }

  function handleDelete() {
    if (isDeleting) return;
    deleteSchedule({ scheduleId: paused.schedule_key });
  }

  return {
    sessionHref,
    handleResume,
    isResuming: isPending,
    canDelete: !paused.routine,
    isDeleteOpen,
    setIsDeleteOpen,
    handleDelete,
    isDeleting,
  };
}
