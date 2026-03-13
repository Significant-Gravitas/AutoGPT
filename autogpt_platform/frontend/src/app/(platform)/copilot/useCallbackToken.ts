import { usePostV2ConsumeCallbackTokenRoute } from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { useEffect, useState } from "react";
import { getGetV2ListSessionsQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";

interface Props {
  isLoggedIn: boolean;
  onConsumed: (sessionId: string) => void;
  onClearAutopilot: () => void;
}

export function useCallbackToken({
  isLoggedIn,
  onConsumed,
  onClearAutopilot,
}: Props) {
  const queryClient = useQueryClient();
  const [callbackToken, setCallbackToken] = useQueryState(
    "callbackToken",
    parseAsString,
  );
  const [consumedTokens, setConsumedTokens] = useState<Set<string>>(
    () => new Set(),
  );
  const { mutateAsync: consumeCallbackToken, isPending } =
    usePostV2ConsumeCallbackTokenRoute();

  const hasConsumedToken =
    callbackToken != null && consumedTokens.has(callbackToken);

  useEffect(() => {
    if (!isLoggedIn || !callbackToken || hasConsumedToken) {
      return;
    }

    let isCancelled = false;
    const token = callbackToken;
    setConsumedTokens((current) => new Set(current).add(token));

    void consumeCallbackToken({ data: { token } })
      .then((response) => {
        if (isCancelled) {
          return;
        }
        if (response.status !== 200 || !response.data?.session_id) {
          throw new Error("Failed to open callback session");
        }

        onConsumed(response.data.session_id);
        onClearAutopilot();
        void setCallbackToken(null);
        queryClient.invalidateQueries({
          queryKey: getGetV2ListSessionsQueryKey(),
        });
      })
      .catch((error) => {
        if (isCancelled) {
          return;
        }
        setConsumedTokens((current) => {
          const next = new Set(current);
          next.delete(token);
          return next;
        });
        void setCallbackToken(null);
        toast({
          title: "Unable to open callback session",
          description:
            error instanceof Error ? error.message : "Please try again.",
          variant: "destructive",
        });
      });

    return () => {
      isCancelled = true;
    };
  }, [
    callbackToken,
    consumeCallbackToken,
    hasConsumedToken,
    isLoggedIn,
    onClearAutopilot,
    onConsumed,
    queryClient,
    setCallbackToken,
  ]);

  return {
    isConsumingCallbackToken: isPending,
  };
}
