"use client";

import {
  getGetV2GetSessionQueryKey,
  usePostV2CancelSessionTask,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { Button } from "@/components/atoms/Button/Button";
import { getTenantRequestInit } from "@/components/contextual/TeamPicker/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useCopilotTenantScope } from "../../CopilotTenantScopeContext";

interface Props {
  sessionId: string;
}

export function BackgroundSubtaskStop({ sessionId }: Props) {
  const queryClient = useQueryClient();
  const scope = useCopilotTenantScope();
  const { mutate, isPending } = usePostV2CancelSessionTask({
    request: getTenantRequestInit(scope.organizationId, scope.teamId),
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
