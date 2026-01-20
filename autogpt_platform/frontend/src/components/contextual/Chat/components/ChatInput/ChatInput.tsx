import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { cn } from "@/lib/utils";
import { ArrowUpIcon } from "@phosphor-icons/react";
import { useChatInput } from "./useChatInput";

export interface Props {
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
}: Props) {
  const inputId = "chat-input";
  const { value, setValue, handleKeyDown, handleSend } = useChatInput({
    onSend,
    disabled,
    maxRows: 5,
    inputId,
  });

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    handleSend();
  }

  return (
    <form onSubmit={handleSubmit} className={cn("relative flex-1", className)}>
      <div className="relative">
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
          wrapperClassName="mb-0"
          className="!rounded-full border-transparent !py-5 pr-12 !text-[1rem] resize-none [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
        />
        <span id="chat-input-hint" className="sr-only">
          Press Enter to send, Shift+Enter for new line
        </span>

        <Button
          type="submit"
          variant="icon"
          size="icon"
          aria-label="Send message"
          className="absolute right-2 top-1/2 -translate-y-1/2 border-zinc-800 bg-zinc-800 text-white hover:border-zinc-900 hover:bg-zinc-900"
          disabled={disabled || !value.trim()}
        >
          <ArrowUpIcon className="h-4 w-4" weight="bold" />
        </Button>
      </div>
    </form>
  );
}
