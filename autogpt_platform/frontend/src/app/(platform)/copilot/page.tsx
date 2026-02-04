"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { Chat } from "@/components/contextual/Chat/Chat";
import { ChatInput } from "@/components/contextual/Chat/components/ChatInput/ChatInput";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useEffect, useState } from "react";
import { useCopilotStore } from "./copilot-page-store";
import { getInputPlaceholder } from "./helpers";
import { useCopilotPage } from "./useCopilotPage";

export default function CopilotPage() {
  const { state, handlers } = useCopilotPage();
  const isInterruptModalOpen = useCopilotStore((s) => s.isInterruptModalOpen);
  const confirmInterrupt = useCopilotStore((s) => s.confirmInterrupt);
  const cancelInterrupt = useCopilotStore((s) => s.cancelInterrupt);

  const [inputPlaceholder, setInputPlaceholder] = useState(
    getInputPlaceholder(),
  );

  useEffect(() => {
    const handleResize = () => {
      setInputPlaceholder(getInputPlaceholder(window.innerWidth));
    };

    handleResize();

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const { greetingName, quickActions, isLoading, hasSession, initialPrompt } =
    state;

  const {
    handleQuickAction,
    startChatWithPrompt,
    handleSessionNotFound,
    handleStreamingChange,
  } = handlers;

  if (hasSession) {
    return (
      <div className="flex h-full flex-col">
        <Chat
          className="flex-1"
          initialPrompt={initialPrompt}
          onSessionNotFound={handleSessionNotFound}
          onStreamingChange={handleStreamingChange}
        />
        <Dialog
          title="Interrupt current chat?"
          styling={{ maxWidth: 300, width: "100%" }}
          controlled={{
            isOpen: isInterruptModalOpen,
            set: (open) => {
              if (!open) cancelInterrupt();
            },
          }}
          onClose={cancelInterrupt}
        >
          <Dialog.Content>
            <div className="flex flex-col gap-4">
              <Text variant="body">
                The current chat response will be interrupted. Are you sure you
                want to continue?
              </Text>
              <Dialog.Footer>
                <Button
                  type="button"
                  variant="outline"
                  onClick={cancelInterrupt}
                >
                  Cancel
                </Button>
                <Button
                  type="button"
                  variant="primary"
                  onClick={confirmInterrupt}
                >
                  Continue
                </Button>
              </Dialog.Footer>
            </div>
          </Dialog.Content>
        </Dialog>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-1 items-center justify-center overflow-y-auto bg-[#f8f8f9] px-3 py-5 md:px-6 md:py-10">
      <div className="w-full text-center">
        {isLoading ? (
          <div className="mx-auto max-w-2xl">
            <Skeleton className="mx-auto mb-3 h-8 w-64" />
            <Skeleton className="mx-auto mb-8 h-6 w-80" />
            <div className="mb-8">
              <Skeleton className="mx-auto h-14 w-full rounded-lg" />
            </div>
            <div className="flex flex-wrap items-center justify-center gap-3">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-9 w-48 rounded-md" />
              ))}
            </div>
          </div>
        ) : (
          <>
            <div className="mx-auto max-w-3xl">
              <Text
                variant="h3"
                className="mb-1 !text-[1.375rem] text-zinc-700"
              >
                Hey, <span className="text-violet-600">{greetingName}</span>
              </Text>
              <Text variant="h3" className="mb-8 !font-normal">
                Tell me about your work — I&apos;ll find what to automate.
              </Text>

              <div className="mb-6">
                <ChatInput
                  onSend={startChatWithPrompt}
                  placeholder={inputPlaceholder}
                />
              </div>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-3 overflow-x-auto [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
              {quickActions.map((action) => (
                <Button
                  key={action}
                  type="button"
                  variant="outline"
                  size="small"
                  onClick={() => handleQuickAction(action)}
                  className="h-auto shrink-0 border-zinc-300 px-3 py-2 text-[.9rem] text-zinc-600"
                >
                  {action}
                </Button>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
