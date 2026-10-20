"use client";

import {
  getGetV2GetSessionQueryKey,
  usePostV2CancelSessionTask,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";

interface Props {
  sessionId: string;
}

export function BackgroundSubtaskStop({ sessionId }: Props) {
  const queryClient = useQueryClient();
  const { mutate, isPending } = usePostV2CancelSessionTask({
    mutation: {
      onSuccess: (response) => {
        if (response.status !== 200 || !response.data.cancelled)
          showStopError();
        return queryClient.invalidateQueries({
          queryKey: getGetV2GetSessionQueryKey(sessionId),
        });
      },
      onError: showStopError,
    },
  });

  function stop() {
    mutate({ sessionId });
  }

  function showStopError() {
    toast({
      title: "Could not stop subtask",
      description: "Please try again.",
      variant: "destructive",
    });
  }

  return (
    <Button
      variant="ghost"
      size="small"
      disabled={isPending}
      onClick={stop}
      aria-label="Stop background subtask"
    >
      {isPending ? "Stopping…" : "Stop"}
    </Button>
  );
}
