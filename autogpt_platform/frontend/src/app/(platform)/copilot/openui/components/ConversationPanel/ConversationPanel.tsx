import type { ComponentProps } from "react";
import { ModeSelector } from "./ModeSelector";
import { ConversationMessages } from "./ConversationMessages";
import { MessageComposer } from "./MessageComposer";

interface Props
  extends ComponentProps<typeof ModeSelector>,
    ComponentProps<typeof ConversationMessages>,
    ComponentProps<typeof MessageComposer> {}

export function ConversationPanel(props: Props) {
  return (
    <section
      className="flex h-full min-h-0 flex-col rounded-2xl border border-zinc-200 bg-white"
      aria-label="Conversation"
    >
      <div className="flex h-14 shrink-0 items-center justify-between border-b border-zinc-100 px-5">
        <span className="text-xs font-semibold text-zinc-800">
          Your conversation
        </span>
        <span className="text-[10px] text-zinc-400">YOU + OTTO</span>
      </div>
      <ModeSelector {...props} />
      <ConversationMessages {...props} />
      <MessageComposer {...props} />
    </section>
  );
}
