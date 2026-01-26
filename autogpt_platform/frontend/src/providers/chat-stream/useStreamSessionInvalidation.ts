import { getGetV2GetSessionQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { useChatStreamStore } from "./chat-stream-store";

export function useStreamSessionInvalidation() {
  const onStreamComplete = useChatStreamStore((state) => state.onStreamComplete);
  const queryClient = useQueryClient();

  useEffect(
    function subscribeToStreamComplete() {
      const unsubscribe = onStreamComplete(function invalidateOnComplete(
        sessionId,
      ) {
        queryClient.invalidateQueries({
          queryKey: getGetV2GetSessionQueryKey(sessionId),
        });
      });

      return unsubscribe;
    },
    [onStreamComplete, queryClient],
  );
}
