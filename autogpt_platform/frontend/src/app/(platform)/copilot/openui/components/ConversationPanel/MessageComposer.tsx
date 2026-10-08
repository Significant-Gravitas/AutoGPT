import type { FormEvent, KeyboardEvent } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { ArrowUp02Icon, StopIcon } from "@hugeicons/core-free-icons";

interface Props {
  prompt: string;
  onPrompt: (value: string) => void;
  isStreaming: boolean;
  suggestions: readonly string[];
  onSend: (message: string) => void;
  onStop: () => void;
  onSubmit: (event: FormEvent) => void;
  onKeyDown: (
    event: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => void;
}

export function MessageComposer(props: Props) {
  return (
    <div className="space-y-3 border-t border-zinc-100 p-4">
      <div className="flex flex-wrap gap-1.5">
        {props.suggestions.map((suggestion) => (
          <Button
            key={suggestion}
            size="xs"
            variant="secondary"
            className="!rounded-full !text-[10px]"
            disabled={props.isStreaming}
            onClick={() => props.onSend(suggestion)}
          >
            {suggestion}
          </Button>
        ))}
      </div>
      <form
        onSubmit={props.onSubmit}
        className="relative rounded-xl border border-zinc-200 bg-zinc-50 focus-within:border-purple-300"
      >
        <Input
          id="openui-message"
          label="Message"
          hideLabel
          type="textarea"
          rows={3}
          value={props.prompt}
          onChange={(event) => props.onPrompt(event.target.value)}
          onKeyDown={props.onKeyDown}
          placeholder="Try a suggested follow-up…"
          maxLength={4000}
          className="!resize-none !border-0 !bg-transparent !pb-12 !text-xs !shadow-none !ring-0"
        />
        <div className="absolute inset-x-3 bottom-2 flex items-center justify-between">
          <span className="text-[10px] text-zinc-400">
            Prepared example · sample data
          </span>
          {props.isStreaming ? (
            <Button
              type="button"
              variant="icon"
              size="icon-xs"
              aria-label="Stop generation"
              onClick={props.onStop}
            >
              <Icon icon={StopIcon} size={13} />
            </Button>
          ) : (
            <Button
              type="submit"
              size="icon-xs"
              aria-label="Send message"
              disabled={!props.prompt.trim()}
            >
              <Icon icon={ArrowUp02Icon} size={16} />
            </Button>
          )}
        </div>
      </form>
    </div>
  );
}
