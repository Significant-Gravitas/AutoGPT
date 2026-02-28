"use client";

import { ChatContainer } from "../ChatContainer/ChatContainer";
import { PaneToolbar } from "./PaneToolbar";
import { useSplitPaneContext } from "./SplitPaneContext";
import { usePaneChat } from "./usePaneChat";

interface Props {
  paneId: string;
  sessionId: string | null;
}

export function ChatPane({ paneId, sessionId: externalSessionId }: Props) {
  const { setPaneSession, setFocusedPaneId } = useSplitPaneContext();

  const {
    sessionId,
    messages,
    status,
    error,
    stop,
    isReconnecting,
    isLoadingSession,
    isSessionError,
    isCreatingSession,
    createSession,
    onSend,
  } = usePaneChat({
    paneId,
    sessionId: externalSessionId,
    onSessionChange: setPaneSession,
  });

  // Derive a title from the first user message
  const firstUserMessage = messages.find((m) => m.role === "user");
  const title = firstUserMessage
    ? firstUserMessage.parts
        .filter((p) => p.type === "text")
        .map((p) => ("text" in p ? p.text : ""))
        .join("")
        .slice(0, 40)
    : null;

  return (
    <div
      className="flex h-full flex-col overflow-hidden"
      onFocus={() => setFocusedPaneId(paneId)}
      onMouseDown={() => setFocusedPaneId(paneId)}
    >
      <PaneToolbar paneId={paneId} title={title} />
      <div className="flex-1 overflow-hidden">
        <ChatContainer
          messages={messages}
          status={status}
          error={error}
          sessionId={sessionId}
          isLoadingSession={isLoadingSession}
          isSessionError={isSessionError}
          isCreatingSession={isCreatingSession}
          isReconnecting={isReconnecting}
          onCreateSession={createSession}
          onSend={onSend}
          onStop={stop}
        />
      </div>
    </div>
  );
}
