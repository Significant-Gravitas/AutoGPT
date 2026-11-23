import { isKey } from "@/lib/keyboard";
import { useRef, useState, type KeyboardEvent } from "react";

interface Args {
  value: string;
  onSave: (text: string) => void;
}

/** Words edited in place: the field holds exactly what is typed; Enter or
 *  leaving the field saves, Shift+Enter starts a new line, Escape keeps
 *  what was there. */
export function useEditableText({ value, onSave }: Args) {
  const [text, setText] = useState<string | null>(null);
  const finished = useRef(false);

  function start() {
    finished.current = false;
    setText(value);
  }

  function finish() {
    finished.current = true;
    setText(null);
  }

  function save() {
    if (finished.current || text === null) return;
    const next = text.trim();
    finish();
    if (next !== value.trim()) onSave(next);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (isKey(e, "Enter") && !e.shiftKey) {
      e.preventDefault();
      save();
      return;
    }
    if (isKey(e, "Escape")) {
      e.preventDefault();
      e.stopPropagation();
      finish();
    }
  }

  return {
    isEditing: text !== null,
    text: text ?? "",
    setText,
    start,
    save,
    onKeyDown,
  };
}
