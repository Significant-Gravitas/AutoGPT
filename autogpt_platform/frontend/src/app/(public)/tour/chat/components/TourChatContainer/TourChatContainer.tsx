"use client";

import { cn } from "@/lib/utils";
import { LightningIcon } from "@phosphor-icons/react";
import type { useTourCopilot } from "../../useTourCopilot";
import { TourMessageList } from "../TourMessageList/TourMessageList";
import { TourPromptBar } from "../TourPromptBar/TourPromptBar";

interface Props {
  chat: ReturnType<typeof useTourCopilot>;
}

export function TourChatContainer({ chat }: Props) {
  return (
    <div className="flex h-full min-h-0 w-full flex-col px-2 lg:px-0">
      {/* Tour-only card styling. These descendant selectors target the shared
          ToolAccordion markup (.bg-stone-50 / .py-2) by class name — if those
          classes change in ToolAccordion, update them here too. Scoped this
          way to avoid modifying the shared component. */}
      <div className="mx-auto flex h-full min-h-0 w-full max-w-3xl flex-col bg-[#fafafa] pb-8 [&_.bg-stone-50]:rounded-xl [&_.bg-stone-50]:border [&_.bg-stone-50]:border-zinc-200/70 [&_.bg-stone-50]:bg-white [&_.bg-stone-50]:!py-2 [&_.bg-stone-50]:shadow-sm [&_.py-2]:py-0">
        <TourMessageList
          messages={chat.messages}
          isStreaming={chat.isStreaming}
        />
        <div
          className={cn(
            "relative px-3 pb-2 pt-2",
            // The upsell renders as a fixed bottom banner once the demo is
            // exhausted — pad so the hint line isn't hidden underneath it.
            chat.isExhausted && "pb-24",
          )}
        >
          <TourPromptBar
            key={chat.turnIndex}
            prompt={chat.currentUserPrompt}
            isStreaming={chat.isStreaming}
            isExhausted={chat.isExhausted}
            onSend={() =>
              chat.currentUserPrompt && chat.onSend(chat.currentUserPrompt)
            }
            onReplay={chat.reset}
          />
          <p className="mt-2 flex items-center justify-center gap-1 text-sm text-zinc-400">
            <LightningIcon className="size-3.5 shrink-0" weight="fill" />
            Simulated demo — pick a scenario above to watch Autopilot build a
            different agent
          </p>
        </div>
      </div>
    </div>
  );
}
