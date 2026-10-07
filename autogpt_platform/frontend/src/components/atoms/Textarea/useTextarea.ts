import { isComposingEvent } from "@/lib/keyboard";
import { useState } from "react";
import { getLength } from "./helpers";

interface Args {
  value: unknown;
  defaultValue: unknown;
  onChange?: React.ChangeEventHandler<HTMLTextAreaElement>;
  onKeyDown?: React.KeyboardEventHandler<HTMLTextAreaElement>;
}

export function useTextarea({
  value,
  defaultValue,
  onChange,
  onKeyDown,
}: Args) {
  const [uncontrolledLength, setUncontrolledLength] = useState(() =>
    getLength(defaultValue),
  );
  const isControlled = value !== undefined;
  const length = isControlled ? getLength(value) : uncontrolledLength;

  function handleChange(event: React.ChangeEvent<HTMLTextAreaElement>) {
    if (!isControlled) setUncontrolledLength(event.target.value.length);
    onChange?.(event);
  }

  // Consumers never see keydowns an IME is still composing; see AGENTS.md
  // "Keyboard handling".
  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (isComposingEvent(event)) return;
    onKeyDown?.(event);
  }

  return {
    length,
    handleChange,
    handleKeyDown: onKeyDown ? handleKeyDown : undefined,
  };
}
