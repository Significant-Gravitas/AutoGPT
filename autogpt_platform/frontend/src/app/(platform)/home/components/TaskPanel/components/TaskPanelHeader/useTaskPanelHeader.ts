import { isKey } from "@/lib/keyboard";
import { useState } from "react";

interface Args {
  title: string;
  onRename: (title: string) => void;
}

export function useTaskPanelHeader({ title, onRename }: Args) {
  const [draft, setDraft] = useState<string | null>(null);

  function startEditing() {
    setDraft(title);
  }

  function stopEditing() {
    setDraft(null);
  }

  function handleChange(
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    setDraft(event.target.value);
  }

  function handleKeyDown(
    event: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    if (isKey(event, "Escape")) {
      event.preventDefault();
      stopEditing();
      return;
    }
    if (!isKey(event, "Enter")) return;
    event.preventDefault();
    const next = draft?.trim() ?? "";
    if (next && next !== title) onRename(next);
    stopEditing();
  }

  return {
    isEditing: draft !== null,
    draft: draft ?? "",
    startEditing,
    stopEditing,
    handleChange,
    handleKeyDown,
  };
}
