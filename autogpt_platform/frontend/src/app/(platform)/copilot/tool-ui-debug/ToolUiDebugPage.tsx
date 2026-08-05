"use client";

import { PlayIcon, RefreshIcon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ChatMessagesContainer } from "../components/ChatMessagesContainer/ChatMessagesContainer";
import { CopilotChatActionsProvider } from "../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { NewChatView } from "./components/NewChatView";
import { type ToolUiVariant, useToolUiDebugPage } from "./useToolUiDebugPage";

function VariantToggle({
  variant,
  onChange,
}: {
  variant: ToolUiVariant;
  onChange: (variant: ToolUiVariant) => void;
}) {
  return (
    <div className="flex items-center rounded-full border border-zinc-200 bg-white p-0.5">
      {(["new", "old"] as const).map((option) => (
        <button
          key={option}
          type="button"
          aria-pressed={variant === option}
          onClick={() => onChange(option)}
          className={
            "rounded-full px-3 py-1 text-xs font-medium capitalize transition-colors " +
            (variant === option
              ? "bg-zinc-900 text-white"
              : "text-zinc-500 hover:text-zinc-800")
          }
        >
          {option}
        </button>
      ))}
    </div>
  );
}

export function ToolUiDebugPage() {
  const {
    messages,
    status,
    isPlaying,
    awaitingUser,
    statusMessage,
    variant,
    setVariant,
    play,
    reset,
    sendUserMessage,
  } = useToolUiDebugPage();

  return (
    <CopilotChatActionsProvider onSend={sendUserMessage}>
      <div className="flex h-[calc(100vh-72px)] flex-col bg-[#fafafa]">
        <header className="flex items-center justify-between border-b border-zinc-200 bg-white px-6 py-3">
          <div className="flex items-center gap-3">
            <h1 className="text-sm font-semibold text-zinc-800">
              Tool UI debug
            </h1>
            <Button
              variant="secondary"
              size="small"
              onClick={play}
              loading={isPlaying}
              leftIcon={<Icon icon={PlayIcon} size={14} />}
            >
              Run sample message
            </Button>
            <Button
              variant="ghost"
              size="small"
              onClick={reset}
              leftIcon={<Icon icon={RefreshIcon} size={14} />}
            >
              Reset
            </Button>
            {awaitingUser && (
              <span className="rounded-full bg-amber-50 px-3 py-1 text-xs font-medium text-amber-600">
                Waiting for your answer — submit the question card to continue
              </span>
            )}
          </div>
          <VariantToggle variant={variant} onChange={setVariant} />
        </header>
        <div className="mx-auto flex min-h-0 w-full max-w-3xl flex-1 flex-col">
          {variant === "old" ? (
            <ChatMessagesContainer
              messages={messages}
              status={status === "streaming" ? "streaming" : "ready"}
              error={undefined}
              isLoading={false}
              readOnly
              forceOldToolUI
            />
          ) : (
            <NewChatView
              messages={messages}
              status={status}
              statusMessage={statusMessage}
            />
          )}
        </div>
      </div>
    </CopilotChatActionsProvider>
  );
}
