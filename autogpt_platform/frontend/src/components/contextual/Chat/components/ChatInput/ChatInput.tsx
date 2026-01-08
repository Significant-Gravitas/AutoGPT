import { Input } from "@/components/atoms/Input/Input";
import { cn } from "@/lib/utils";
import { ArrowUpIcon } from "@phosphor-icons/react";
import { useChatInput } from "./useChatInput";

export interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

export function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Type your message...",
  className,
}: ChatInputProps) {
  const inputId = "chat-input";
  const { value, setValue, handleKeyDown, handleSend } = useChatInput({
    onSend,
    disabled,
    maxRows: 5,
    inputId,
  });

  return (
    <div className={cn("relative flex-1", className)}>
      <Input
        id={inputId}
        label="Chat message input"
        hideLabel
        type="textarea"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        rows={1}
        wrapperClassName="mb-0 relative"
        className="pr-12"
      />
      <span id="chat-input-hint" className="sr-only">
        Press Enter to send, Shift+Enter for new line
      </span>

      <button
        onClick={handleSend}
        disabled={disabled || !value.trim()}
        className={cn(
          "absolute right-3 top-1/2 flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-full",
          "border border-zinc-800 bg-zinc-800 text-white",
          "hover:border-zinc-900 hover:bg-zinc-900",
          "disabled:border-zinc-200 disabled:bg-zinc-200 disabled:text-white disabled:opacity-50",
          "transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-neutral-950",
          "disabled:pointer-events-none",
        )}
        aria-label="Send message"
      >
        <ArrowUpIcon className="h-3 w-3" weight="bold" />
      </button>
    </div>
  );
}
