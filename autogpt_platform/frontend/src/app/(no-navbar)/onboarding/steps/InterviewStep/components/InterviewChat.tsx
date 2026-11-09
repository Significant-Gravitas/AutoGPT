"use client";

import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { ChatMessagesContainer } from "@/app/(platform)/copilot/components/ChatMessagesContainer/ChatMessagesContainer";
import { CopilotChatActionsProvider } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

type ChatProps = React.ComponentProps<typeof ChatMessagesContainer>;

interface Props {
  sessionId: string;
  messages: ChatProps["messages"];
  status: ChatProps["status"];
  error: ChatProps["error"];
  queuedMessages: ChatProps["queuedMessages"];
  isHydrating: boolean;
  isTurnSettling: boolean;
  isTurnRunning: boolean;
  isOver: boolean;
  canFinishEarly: boolean;
  isLeaving: boolean;
  hasFinishFailed: boolean;
  onSend: (message: string) => Promise<void>;
  onStop: () => void;
  onFinish: () => Promise<void>;
}

export function InterviewChat({
  sessionId,
  messages,
  status,
  error,
  queuedMessages,
  isHydrating,
  isTurnSettling,
  isTurnRunning,
  isOver,
  canFinishEarly,
  isLeaving,
  hasFinishFailed,
  onSend,
  onStop,
  onFinish,
}: Props) {
  return (
    <CopilotChatActionsProvider onSend={onSend}>
      <div
        role="region"
        aria-label="Interview with Otto"
        className="flex h-[60vh] max-h-[40rem] min-h-[20rem] w-full flex-col overflow-hidden rounded-xl border border-border-base bg-surface"
      >
        <div className="flex min-h-0 flex-1 flex-col">
          <ChatMessagesContainer
            messages={messages}
            status={status}
            error={error}
            isLoading={isHydrating}
            sessionID={sessionId}
            queuedMessages={queuedMessages}
            isTurnSettling={isTurnSettling}
            variant="compact"
          />
        </div>
        <div className="flex shrink-0 flex-col gap-2 border-t border-separator px-3 py-3">
          {isOver ? (
            <div className="flex justify-center">
              <Button
                variant="primary"
                size="large"
                loading={isLeaving}
                disabled={isLeaving}
                onClick={() => void onFinish()}
              >
                Meet your team
              </Button>
            </div>
          ) : (
            <div className="flex items-end gap-2">
              <div className="min-w-0 flex-1">
                <ChatInput
                  inputId="onboarding-interview-input"
                  onSend={onSend}
                  disabled={isLeaving}
                  isStreaming={isTurnRunning}
                  isTurnStarting={status === "submitted"}
                  onStop={onStop}
                  onEnqueue={onSend}
                  placeholder="Type your answer…"
                  hasSession={true}
                  variant="compact"
                />
              </div>
              {canFinishEarly && (
                <Button
                  variant="secondary"
                  size="small"
                  loading={isLeaving}
                  disabled={isLeaving}
                  onClick={() => void onFinish()}
                >
                  I&apos;m done
                </Button>
              )}
            </div>
          )}
          {hasFinishFailed && (
            <Text variant="small" tone="error" role="alert">
              That didn&apos;t go through. Try again.
            </Text>
          )}
        </div>
      </div>
    </CopilotChatActionsProvider>
  );
}
