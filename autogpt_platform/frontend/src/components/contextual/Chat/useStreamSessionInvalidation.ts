"use client";

import { getGetV2GetSessionQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { useChatStore } from "./chat-store";

export function useStreamSessionInvalidation() {
  const onStreamComplete = useChatStore((state) => state.onStreamComplete);
  const queryClient = useQueryClient();

  useEffect(
    function subscribeToStreamComplete() {
      const unsubscribe = onStreamComplete(
        function invalidateOnComplete(sessionId) {
          queryClient.invalidateQueries({
            queryKey: getGetV2GetSessionQueryKey(sessionId),
          });
        },
      );

      return unsubscribe;
    },
    [onStreamComplete, queryClient],
  );
}
