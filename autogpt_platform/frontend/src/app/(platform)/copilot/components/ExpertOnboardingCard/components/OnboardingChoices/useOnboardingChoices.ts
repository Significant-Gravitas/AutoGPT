import { isComposingEvent, isKey } from "@/lib/keyboard";
import {
  useImperativeHandle,
  useRef,
  useState,
  type KeyboardEvent,
  type Ref,
} from "react";
import {
  toAnswerText,
  toMultiAnswer,
  type QuestionAnswer,
} from "../../../../tools/clarifying-questions";
import { getLetterIndex } from "../../helpers";

export interface OnboardingChoicesHandle {
  /** Picks the row a bare letter key names. Returns whether it did. */
  pickLetter: (event: KeyboardEvent) => boolean;
}

interface Args {
  ref?: Ref<OnboardingChoicesHandle>;
  options: string[];
  answer: QuestionAnswer | undefined;
  isMultiple: boolean;
  onChoose: (option: string) => void;
  onChange: (answer: QuestionAnswer) => void;
  onSubmit: () => void;
}

export function useOnboardingChoices({
  ref,
  options,
  answer,
  isMultiple,
  onChoose,
  onChange,
  onSubmit,
}: Args) {
  // A single answer is one string, a tapped option or typed text. Several
  // answers keep the ticks and the typed text apart, so text that happens to
  // match an option is still the user's own words.
  const multi = toMultiAnswer(answer);
  const text = isMultiple ? multi.custom : toAnswerText(answer);
  const trimmed = text.trim();
  const selected = isMultiple
    ? multi.selected.filter((option) => options.includes(option))
    : options.filter((option) => option === trimmed);
  const isCustom =
    trimmed.length > 0 && (isMultiple || !options.includes(trimmed));
  const [isTyping, setIsTyping] = useState(isCustom || options.length === 0);
  const [active, setActive] = useState(() =>
    Math.max(options.indexOf(selected[0] ?? ""), 0),
  );
  const optionRefs = useRef<(HTMLButtonElement | null)[]>([]);

  useImperativeHandle(ref, () => ({ pickLetter }));

  function toggle(option: string) {
    const next = selected.includes(option)
      ? selected.filter((pick) => pick !== option)
      : [...selected, option];
    // Kept in the order offered, so the reply reads like the question.
    onChange({
      selected: options.filter((pick) => next.includes(pick)),
      custom: text,
    });
  }

  function choose(option: string) {
    setActive(options.indexOf(option));
    if (isMultiple) {
      toggle(option);
      return;
    }
    setIsTyping(false);
    onChoose(option);
  }

  function startTyping() {
    if (!isMultiple && !isCustom) onChange("");
    setIsTyping(true);
  }

  function changeText(next: string) {
    onChange(isMultiple ? { selected, custom: next } : next);
  }

  // Arrows only move: picking with them must not jump to the next question
  // while the user is still browsing the list, or tick boxes on the way past.
  function moveTo(index: number) {
    setActive(index);
    if (!isMultiple) onChange(options[index]);
    optionRefs.current[index]?.focus();
  }

  // A letter picks its row, the last one being "Other".
  function pickLetter(event: KeyboardEvent) {
    if (isComposingEvent(event)) return false;
    const letter = getLetterIndex(event);
    if (letter === null || letter > options.length) return false;
    event.preventDefault();
    if (letter === options.length) startTyping();
    else choose(options[letter]);
    return true;
  }

  function handleOptionKeyDown(event: KeyboardEvent, index: number) {
    const option = options[index];
    if (isKey(event, "Enter")) {
      event.preventDefault();
      const isPicked = selected.includes(option);
      // With several answers, Enter is "done ticking" once any is ticked.
      if (isPicked || (isMultiple && selected.length > 0)) onSubmit();
      else choose(option);
      return;
    }
    if (isKey(event, " ")) {
      event.preventDefault();
      choose(option);
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
    isTyping,
    optionRefs,
    selected,
    text,
    changeText,
    choose,
    startTyping,
    handleOptionKeyDown,
    handleTextKeyDown,
  };
}
