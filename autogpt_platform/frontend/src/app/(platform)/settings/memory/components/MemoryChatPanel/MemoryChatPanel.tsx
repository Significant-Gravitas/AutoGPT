"use client";

import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { ChatMessagesContainer } from "@/app/(platform)/copilot/components/ChatMessagesContainer/ChatMessagesContainer";
import { CopilotChatActionsProvider } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { BrainIcon, Cancel01Icon } from "@hugeicons/core-free-icons";

interface Props {
  scopeName: string;
  isOpen: boolean;
  isStarting: boolean;
  startError: boolean;
  sessionId: string | null;
  messages: React.ComponentProps<typeof ChatMessagesContainer>["messages"];
  status: React.ComponentProps<typeof ChatMessagesContainer>["status"];
  error: React.ComponentProps<typeof ChatMessagesContainer>["error"];
  queuedMessages: React.ComponentProps<
    typeof ChatMessagesContainer
  >["queuedMessages"];
  onSend: (message: string) => Promise<void>;
  onStop: () => void;
  onRetry: () => void;
  onClose: () => void;
}

export function MemoryChatPanel({
  scopeName,
  isOpen,
  isStarting,
  startError,
  sessionId,
  messages,
  status,
  error,
  queuedMessages,
  onSend,
  onStop,
  onRetry,
  onClose,
}: Props) {
  if (!isOpen) return null;

  const isStreaming = status === "streaming" || status === "submitted";

  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex flex-col items-end">
      <CopilotChatActionsProvider onSend={onSend}>
        <div
          role="complementary"
          aria-label="Memory chat panel"
          className="pointer-events-auto flex h-[70vh] max-h-[calc(100vh-6rem)] w-[26rem] max-w-[calc(100vw-2rem)] flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-2xl sm:h-[75vh]"
        >
          <div className="flex items-center justify-between border-b border-zinc-100 px-4 py-3">
            <div className="flex items-center gap-2">
              <Icon icon={BrainIcon} size={18} className="text-violet-600" />
              <span className="text-sm font-semibold text-zinc-800">
                {scopeName}&apos;s memory
              </span>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              aria-label="Close memory chat"
            >
              <Icon icon={Cancel01Icon} size={16} />
            </Button>
          </div>

          <div className="flex h-0 min-h-0 flex-1 flex-col">
            {startError ? (
              <div className="flex flex-1 flex-col items-center justify-center gap-3 px-4 py-6 text-center text-sm text-zinc-600">
                <p className="font-medium text-zinc-800">
                  Could not start the memory chat
                </p>
                <p className="text-zinc-500">
                  Something went wrong. Retry to try again.
                </p>
                <Button variant="secondary" size="small" onClick={onRetry}>
                  Retry
                </Button>
              </div>
            ) : isStarting || !sessionId ? (
              <div className="flex flex-1 items-center justify-center px-4 py-6 text-sm text-zinc-500">
                Starting memory chat…
              </div>
            ) : (
              <>
                <div className="flex min-h-0 flex-1 flex-col">
                  <ChatMessagesContainer
                    messages={messages}
                    status={status}
                    error={error}
                    isLoading={false}
                    sessionID={sessionId}
                    queuedMessages={queuedMessages}
                  />
                </div>
                <div className="relative shrink-0 border-t border-zinc-100 bg-white px-3 pb-2 pt-2">
                  <ChatInput
                    inputId="memory-chat-input"
                    onSend={onSend}
                    disabled={false}
                    isStreaming={isStreaming}
                    onStop={onStop}
                    onEnqueue={onSend}
                    placeholder="Ask or update…"
                    hasSession={true}
                  />
                </div>
              </>
            )}
          </div>
        </div>
      </CopilotChatActionsProvider>
    </div>
  );
}
