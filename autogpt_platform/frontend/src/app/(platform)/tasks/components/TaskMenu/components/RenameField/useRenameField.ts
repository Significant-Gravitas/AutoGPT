import { isKey } from "@/lib/keyboard";
import { useRef, useState, type KeyboardEvent } from "react";
import { useRenameTask } from "../../../../data/useTasksData";

interface Args {
  taskId: string;
  title: string;
  onDone: () => void;
}

export function useRenameField({ taskId, title, onDone }: Args) {
  const [text, setText] = useState(title);
  const rename = useRenameTask();
  const finished = useRef(false);

  function finish() {
    if (finished.current) return;
    finished.current = true;
    onDone();
  }

  function save() {
    if (finished.current) return;
    const next = text.trim();
    if (next && next !== title) rename.mutate({ id: taskId, title: next });
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
