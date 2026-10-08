import { AGENT_STATUS_LABEL } from "@/components/molecules/AgentStatusAvatar/helpers";
import { usePendingReviewsForChatSession } from "@/hooks/usePendingReviews";
import { useRouter } from "next/navigation";
import { parseAsString, useQueryState } from "nuqs";
import { useCopilotPage } from "../useCopilotPage";
import { useSessionList } from "../useSessionList";
import { TECHNICAL_VIEW_PARAM } from "./useCompactModeRedirect";
import {
  deriveAgentStatus,
  describeAgentActivity,
  getVisibleMessages,
} from "./helpers";

const RECENT_SESSION_LIMIT = 20;

export function useCompactCopilotPage() {
  const chat = useCopilotPage();
  const router = useRouter();
  const [, setSessionId] = useQueryState("sessionId", parseAsString);
  const { sessions } = useSessionList();
  const { pendingReviews } = usePendingReviewsForChatSession(
    chat.sessionId ?? "",
    { enabled: !!chat.sessionId },
  );

  const messages = getVisibleMessages(chat.messages);
  const agentStatus = deriveAgentStatus({
    status: chat.status,
    messages,
    hasPendingReviews: pendingReviews.length > 0,
    isReconnecting: chat.isReconnecting,
  });
  const isBusy = chat.status === "submitted" || chat.status === "streaming";

  function openSession(id: string | null) {
    void setSessionId(id);
  }

  function openDetailedView() {
    const params = new URLSearchParams({ [TECHNICAL_VIEW_PARAM]: "technical" });
    if (chat.sessionId) params.set("sessionId", chat.sessionId);
    router.push(`/home?${params.toString()}`);
  }

  return {
    sessionId: chat.sessionId,
    messages,
    rawMessages: chat.messages,
    agentStatus,
    agentActivity:
      describeAgentActivity(agentStatus, messages) ??
      AGENT_STATUS_LABEL[agentStatus],
    isBusy,
    isStreaming: chat.status === "streaming",
    isLoadingSession: chat.isLoadingSession,
    isSessionError: chat.isSessionError || chat.isSessionNotFound,
    isSending: chat.isCreatingSession || chat.isUploadingFiles,
    queuedMessages: chat.queuedMessages,
    error: chat.error,
    sessions: sessions.slice(0, RECENT_SESSION_LIMIT),
    onSend: chat.onSend,
    onStop: chat.stop,
    onBackendTurn: chat.followBackendTurn,
    openSession,
    openDetailedView,
  };
}
