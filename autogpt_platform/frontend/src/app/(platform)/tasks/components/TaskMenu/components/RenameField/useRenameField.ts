import { isKey } from "@/lib/keyboard";
import { useRef, useState, type KeyboardEvent } from "react";
import { useTaskActions } from "../../../../useTaskActions";

interface Args {
  taskID: string;
  title: string;
  onDone: () => void;
}

export function useRenameField({ taskID, title, onDone }: Args) {
  const [text, setText] = useState(title);
  const actions = useTaskActions(taskID);
  const finished = useRef(false);

  function finish() {
    if (finished.current) return;
    finished.current = true;
    onDone();
  }

  function save() {
    if (finished.current) return;
    const next = text.trim();
    if (next && next !== title) actions.rename(next);
    finish();
  }

  function onKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (isKey(e, "Enter")) {
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

  return { text, setText, save, onKeyDown };
}
