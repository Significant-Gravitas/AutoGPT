"use client";
import { useTourStore } from "../../tourStore";
import type { useTourCopilot } from "../../useTourCopilot";
import { TourEndCard } from "../TourEndCard/TourEndCard";
import { TourMessageList } from "../TourMessageList/TourMessageList";
import { TourPromptBar } from "../TourPromptBar/TourPromptBar";
import { FlashIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  chat: ReturnType<typeof useTourCopilot>;
}

export function TourChatContainer({ chat }: Props) {
  const isDemoComplete = useTourStore((s) => s.isDemoComplete);

  return (
    <div className="flex h-full min-h-0 w-full flex-col px-2 lg:px-0">
      <div className="mx-auto flex h-full min-h-0 w-full max-w-3xl flex-col bg-background pb-8">
        <TourMessageList
          messages={chat.messages}
          isStreaming={chat.isStreaming}
          footer={isDemoComplete ? <TourEndCard /> : null}
        />
        {!isDemoComplete && (
          <div className="relative px-3 pt-2 pb-2">
            <TourPromptBar
              key={`${chat.turnIndex}:${chat.currentUserPrompt ?? ""}`}
              prompt={chat.currentUserPrompt}
              isStreaming={chat.isStreaming}
              onSend={() =>
                chat.currentUserPrompt && chat.onSend(chat.currentUserPrompt)
              }
            />
            <Text
              variant="body"
              className="mt-2 flex items-center justify-center gap-1 text-zinc-400"
            >
              <Icon icon={FlashIcon} className="size-3.5 shrink-0" />
              Simulated demo — sign up to put Otto to work on your own tasks.
            </Text>
          </div>
        )}
      </div>
    </div>
  );
}
