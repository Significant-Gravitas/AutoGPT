import { isComposingEvent, isKey } from "@/lib/keyboard";
import { useRef, useState, type KeyboardEvent } from "react";
import { getLetterIndex } from "../../helpers";

interface Args {
  options: string[];
  value: string;
  onChoose: (option: string) => void;
  onChange: (value: string) => void;
  onSubmit: () => void;
}

export function useOnboardingChoices({
  options,
  value,
  onChoose,
  onChange,
  onSubmit,
}: Args) {
  const trimmed = value.trim();
  const isCustom = trimmed.length > 0 && !options.includes(trimmed);
  const [isTyping, setIsTyping] = useState(isCustom || options.length === 0);
  // A saved custom answer opens the text box on mount, but only a click on
  // Other may pull focus into it.
  const [didOpenOther, setDidOpenOther] = useState(false);
  const optionRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const selected = options.indexOf(trimmed);
  const active = selected === -1 ? 0 : selected;

  function choose(option: string) {
    setIsTyping(false);
    onChoose(option);
  }

  function startTyping() {
    if (!isCustom) onChange("");
    setIsTyping(true);
    setDidOpenOther(true);
  }

  // Arrows only move the selection: picking with them must not jump to the
  // next question while the user is still browsing the list.
  function moveTo(index: number) {
    onChange(options[index]);
    optionRefs.current[index]?.focus();
  }

  // A letter picks its row from anywhere in the list, the last one being
  // "Other". Returns whether the key was used.
  function handleLetterKey(event: KeyboardEvent) {
    if (isComposingEvent(event)) return false;
    const letter = getLetterIndex(event);
    if (letter === null || letter > options.length) return false;
    event.preventDefault();
    if (letter === options.length) startTyping();
    else choose(options[letter]);
    return true;
  }

  function handleOptionKeyDown(event: KeyboardEvent, index: number) {
    if (handleLetterKey(event)) return;
    if (isKey(event, "Enter")) {
      event.preventDefault();
      if (options[index] === trimmed) onSubmit();
      else choose(options[index]);
      return;
    }
    if (isKey(event, " ")) {
      event.preventDefault();
      choose(options[index]);
      return;
    }
    const step = isKey(event, "ArrowDown", "ArrowRight")
      ? 1
      : isKey(event, "ArrowUp", "ArrowLeft")
        ? -1
        : 0;
    if (step === 0) return;
    event.preventDefault();
    moveTo((index + step + options.length) % options.length);
  }

  // Enter moves on; Shift+Enter is the newline.
  function handleTextKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (!isKey(event, "Enter") || event.shiftKey) return;
    event.preventDefault();
    onSubmit();
  }

  return {
    active,
    didOpenOther,
    isTyping,
    optionRefs,
    choose,
    startTyping,
    handleLetterKey,
    handleOptionKeyDown,
    handleTextKeyDown,
  };
}
