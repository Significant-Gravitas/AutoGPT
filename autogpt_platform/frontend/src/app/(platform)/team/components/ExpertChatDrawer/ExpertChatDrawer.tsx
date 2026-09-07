"use client";

import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { ChatMessagesContainer } from "@/app/(platform)/copilot/components/ChatMessagesContainer/ChatMessagesContainer";
import { CopilotChatActionsProvider } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowExpandIcon, NoteEditIcon } from "@hugeicons/core-free-icons";
import { ExpertSidePanel } from "../ExpertSidePanel/ExpertSidePanel";
import {
  IdentityAvatar,
  PanelIdentity,
} from "../ExpertSidePanel/IdentityAvatar";
import type { ChatTarget } from "./helpers";
import { useExpertChatDrawer } from "./useExpertChatDrawer";

const DEFAULT_CHAT_WIDTH = 480;

interface Props {
  target: ChatTarget | null;
  onClose: () => void;
  resumeLatest?: boolean;
  threadKey?: number;
  /** Sent as the first message of the thread started by `threadKey`. */
  seedPrompt?: string | null;
}

export function ExpertChatDrawer({
  target,
  onClose,
  resumeLatest = true,
  threadKey = 0,
  seedPrompt = null,
}: Props) {
  const chat = useExpertChatDrawer({
    target,
    isOpen: target !== null,
    resumeLatest,
    threadKey,
    seedPrompt,
  });
  const copilotHref = chat.sessionId
    ? `/copilot?sessionId=${encodeURIComponent(chat.sessionId)}`
    : target?.expertId
      ? `/copilot?expertId=${target.expertId}&new=1`
      : "/copilot";
  const identity = target
    ? {
        name: target.name,
        avatarUrl: target.avatarUrl,
        isAutopilot: target.expertId === null,
      }
    : null;

  return (
    <ExpertSidePanel
      identity={identity}
      title={target?.name ?? ""}
      ariaLabel={target ? `Chat with ${target.name}` : ""}
      panelId="chat"
      closeLabel="Close chat panel"
      showIdentity={!!chat.sessionId}
      headerActions={
        <>
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            leadingIcon={NoteEditIcon}
            aria-label="New task"
            disabled={!chat.sessionId}
            onClick={chat.startNewThread}
          />
          <Button
            as="NextLink"
            href={copilotHref}
            variant="ghost"
            size="icon-xs"
            leadingIcon={ArrowExpandIcon}
            aria-label="Open in Copilot"
          />
        </>
      }
      defaultWidth={DEFAULT_CHAT_WIDTH}
      onClose={onClose}
    >
      {identity && target ? (
        <ChatPanelBody target={target} identity={identity} chat={chat} />
      ) : null}
    </ExpertSidePanel>
  );
}

interface BodyProps {
  target: ChatTarget;
  identity: PanelIdentity;
  chat: ReturnType<typeof useExpertChatDrawer>;
}

function ChatPanelBody({ target, identity, chat }: BodyProps) {
  const {
    sessionId,
    messages,
    status,
    error,
    stop,
    onSend,
    queuedMessages,
    isResolvingSession,
    isCreating,
  } = chat;

  const isStreaming = status === "streaming" || status === "submitted";

  return (
    <CopilotChatActionsProvider onSend={onSend}>
      <div className="flex min-h-0 flex-1 flex-col">
        {isResolvingSession ? (
          <div className="flex flex-1 items-center justify-center px-4 py-6">
            <Text variant="small" tone="muted">
              Opening chat…
            </Text>
          </div>
        ) : sessionId ? (
          <div className="flex min-h-0 flex-1 flex-col">
            <ChatMessagesContainer
              messages={messages}
              status={status}
              error={error}
              isLoading={false}
              sessionID={sessionId}
              queuedMessages={queuedMessages}
              variant="compact"
              showThreadHeader={false}
            />
          </div>
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 py-6 text-center">
            <IdentityAvatar
              identity={identity}
              className="h-24 w-24"
              imageSize={192}
            />
            <div className="space-y-0.5">
              <Text variant="body-medium" tone="primary">
                What can I do for you?
              </Text>
              <Text variant="small" tone="muted">
                {target.name} · {target.role}
              </Text>
            </div>
          </div>
        )}
        <div className="shrink-0 px-3 pb-5 pt-2">
          <ChatInput
            inputId="expert-chat-input"
            variant="compact"
            onSend={onSend}
            disabled={isResolvingSession || isCreating}
            isStreaming={isStreaming}
            onStop={stop}
            onEnqueue={onSend}
            placeholder={`Message ${target.name}…`}
            hasSession={!!sessionId}
          />
        </div>
      </div>
    </CopilotChatActionsProvider>
  );
}
