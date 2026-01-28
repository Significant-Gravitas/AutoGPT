import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { ArrowUpIcon, StopIcon } from "@phosphor-icons/react";
import { useChatInput } from "./useChatInput";

export interface Props {
  onSend: (message: string) => void;
  disabled?: boolean;
  isStreaming?: boolean;
  onStop?: () => void;
  placeholder?: string;
  className?: string;
}

export function ChatInput({
  onSend,
  disabled = false,
  isStreaming = false,
  onStop,
  placeholder = "Type your message...",
  className,
}: Props) {
  const inputId = "chat-input";
  const { value, handleKeyDown, handleSubmit, handleChange, hasMultipleLines } =
    useChatInput({
      onSend,
      disabled: disabled || isStreaming,
      maxRows: 4,
      inputId,
    });

  return (
    <form onSubmit={handleSubmit} className={cn("relative flex-1", className)}>
      <div className="relative">
        <div
          id={`${inputId}-wrapper`}
          className={cn(
            "relative overflow-hidden border border-neutral-200 bg-white shadow-sm",
            "focus-within:border-zinc-400 focus-within:ring-1 focus-within:ring-zinc-400",
            hasMultipleLines ? "rounded-xlarge" : "rounded-full",
          )}
        >
          <textarea
            id={inputId}
            aria-label="Chat message input"
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled || isStreaming}
            rows={1}
            className={cn(
              "w-full resize-none overflow-y-auto border-0 bg-transparent text-[1rem] leading-6 text-black",
              "placeholder:text-zinc-400",
              "focus:outline-none focus:ring-0",
              "disabled:text-zinc-500",
              hasMultipleLines ? "pb-6 pl-4 pr-4 pt-2" : "pb-4 pl-4 pr-14 pt-4",
            )}
          />
        </div>
        <span id="chat-input-hint" className="sr-only">
          Press Enter to send, Shift+Enter for new line
        </span>

        {isStreaming ? (
          <Button
            type="button"
            variant="icon"
            size="icon"
            aria-label="Stop generating"
            onClick={onStop}
            className="absolute bottom-[7px] right-2 border-red-600 bg-red-600 text-white hover:border-red-800 hover:bg-red-800"
          >
            <StopIcon className="h-4 w-4" weight="bold" />
          </Button>
        ) : (
          <Button
            type="submit"
            variant="icon"
            size="icon"
            aria-label="Send message"
            className={cn(
              "absolute bottom-[7px] right-2 border-zinc-800 bg-zinc-800 text-white hover:border-zinc-900 hover:bg-zinc-900",
              (disabled || !value.trim()) && "opacity-20",
            )}
            disabled={disabled || !value.trim()}
          >
            <ArrowUpIcon className="h-4 w-4" weight="bold" />
          </Button>
        )}
      </div>
    </form>
  );
}
