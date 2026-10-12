import { useGetV2ListChatTransports } from "@/app/api/__generated__/endpoints/chat/chat";
import { getAvailableLLMTransports } from "@/app/(platform)/copilot/helpers/copilotLlmAuth";

export function useNoAIConnectionNotice(enabled: boolean) {
  const transportsQuery = useGetV2ListChatTransports({
    query: { enabled },
  });
  const transports =
    transportsQuery.data?.status === 200
      ? transportsQuery.data.data.transports
      : undefined;

  return {
    hasNoConnection:
      enabled &&
      transports !== undefined &&
      getAvailableLLMTransports(transports).length === 0,
  };
}
