"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Textarea } from "@/components/atoms/Textarea/Textarea";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { ArrowUp02Icon, StopIcon } from "@hugeicons/core-free-icons";
import { useCompactComposer } from "./useCompactComposer";

interface Props {
  onSend: (text: string) => Promise<void>;
  onStop: () => void;
  isBusy: boolean;
  disabled?: boolean;
}

export function CompactComposer({ onSend, onStop, isBusy, disabled }: Props) {
  const { value, setValue, canSend, submit, handleKeyDown } =
    useCompactComposer({ onSend });
  const showStop = isBusy && !canSend;

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submit();
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="mx-auto flex w-full max-w-2xl shrink-0 items-end gap-2 px-4 pt-2 pb-[max(1rem,env(safe-area-inset-bottom))]"
    >
      <Textarea
        label={`Message ${AUTOPILOT_NAME}`}
        hideLabel
        rows={1}
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={`Message ${AUTOPILOT_NAME}`}
        disabled={disabled}
        wrapperClassName="min-w-0 flex-1"
        className="max-h-40 resize-none rounded-2xl bg-card"
      />
      {showStop ? (
        <Button
          type="button"
          variant="secondary"
          size="icon-lg"
          aria-label="Stop"
          withTooltip={false}
          onClick={onStop}
        >
          <Icon icon={StopIcon} size={18} />
        </Button>
      ) : (
        <Button
          type="submit"
          variant="primary"
          size="icon-lg"
          aria-label="Send"
          withTooltip={false}
          disabled={!canSend || disabled}
        >
          <Icon icon={ArrowUp02Icon} size={18} />
        </Button>
      )}
    </form>
  );
}
