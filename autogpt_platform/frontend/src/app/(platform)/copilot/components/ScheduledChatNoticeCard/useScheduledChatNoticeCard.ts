import { getGetV2GetSessionQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import { getGetHomeDashboardQueryKey } from "@/app/api/__generated__/endpoints/home/home";
import {
  getListPausedScheduledChatsQueryKey,
  useResumeScheduledChat,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import { toast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { invalidateAllScheduleQueries } from "@/services/schedules/invalidate-schedules";
import * as Sentry from "@sentry/nextjs";
import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";

interface Args {
  scheduleKey: string | null;
  sessionID?: string | null;
}

export function useScheduledChatNoticeCard({ scheduleKey, sessionID }: Args) {
  const queryClient = useQueryClient();
  const [connectProvider, setConnectProvider] = useState<string | null>(null);
  const [resumed, setResumed] = useState(false);
  const [lessOftenOpen, setLessOftenOpen] = useState(false);
  // A resume that follows a reconnect is quiet when there was nothing to
  // resume: a chat that was only warned, never paused, answers 404.
  const quietRef = useRef(false);

  function refresh() {
    queryClient.invalidateQueries({
      queryKey: getListPausedScheduledChatsQueryKey(),
    });
    // Home lists each paused chat under Needs you.
    queryClient.invalidateQueries({ queryKey: getGetHomeDashboardQueryKey() });
    void invalidateAllScheduleQueries(queryClient);
    if (sessionID) {
      queryClient.invalidateQueries({
        queryKey: getGetV2GetSessionQueryKey(sessionID),
      });
    }
  }

  const { mutate, isPending } = useResumeScheduledChat({
    mutation: {
      onSuccess: () => {
        setResumed(true);
        refresh();
        toast({
          title: "Resumed",
          description: "It runs again at its next scheduled time.",
        });
      },
      onError: (error) => {
        if (error instanceof ApiError && error.status === 404) {
          setResumed(true);
          refresh();
          if (!quietRef.current) toast({ title: "Already running" });
          return;
        }
        Sentry.captureException(error);
        toast({
          variant: "destructive",
          title: "Couldn't resume this scheduled chat",
          description: "Please try again.",
        });
      },
    },
  });

  function resume({ quiet = false } = {}) {
    if (!scheduleKey || isPending || resumed) return;
    quietRef.current = quiet;
    mutate({ scheduleKey });
  }

  function handleConnected() {
    setConnectProvider(null);
    // Reconnecting is the fix, so it resumes the chat too.
    resume({ quiet: true });
  }

  return {
    connectProvider,
    openConnect: setConnectProvider,
    closeConnect: () => setConnectProvider(null),
    handleConnected,
    resume: () => resume(),
    isResuming: isPending,
    resumed,
    lessOftenOpen,
    setLessOftenOpen,
  };
}
