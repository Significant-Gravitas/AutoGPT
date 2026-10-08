import { useEffect, useRef } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { AiMagicIcon } from "@hugeicons/core-free-icons";
import type { LabMessage } from "../../useOpenUILab";
import { cn } from "@/lib/utils";

interface Props {
  messages: LabMessage[];
  isStreaming: boolean;
}

function Message({ message }: { message: LabMessage }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
        {message.role === "assistant" && (
          <Icon icon={AiMagicIcon} size={14} className="text-purple-500" />
        )}
        {message.role === "user" ? "You" : "Otto"}
      </div>
      <p
        className={cn(
          "text-sm leading-relaxed text-zinc-600",
          message.role === "user" && "font-medium text-zinc-800",
        )}
      >
        {message.text}
      </p>
      {message.role === "assistant" && (
        <span className="inline-flex items-center gap-1.5 rounded-md bg-purple-50 px-2 py-1 text-[10px] font-medium text-purple-600">
          <span className="size-1 rounded-full bg-purple-400" />
          Interactive workspace
        </span>
      )}
    </div>
  );
}

export function ConversationMessages({ messages, isStreaming }: Props) {
  const log = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (log.current)
      log.current.scrollTop =
        messages.length > 2 ? log.current.scrollHeight : 0;
  }, [messages, isStreaming]);
  return (
    <div
      ref={log}
      className="min-h-0 flex-1 space-y-6 overflow-y-auto p-5"
      role="log"
      aria-label="Conversation messages"
      aria-live="polite"
    >
      {messages.map((message, index) => (
        <Message key={index} message={message} />
      ))}
      {isStreaming && (
        <p
          role="status"
          className="text-xs text-purple-600 motion-safe:animate-pulse"
        >
          Rendering the sample…
        </p>
      )}
    </div>
  );
}
