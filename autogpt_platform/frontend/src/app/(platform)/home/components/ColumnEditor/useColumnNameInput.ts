import { isKey } from "@/lib/keyboard";
import { useState } from "react";
import { MAX_COLUMN_NAME } from "../../board-helpers";

interface Args {
  name: string;
  onSave: (name: string) => void;
  onCancel: () => void;
}

export function useColumnNameInput({ name, onSave, onCancel }: Args) {
  const [draft, setDraft] = useState(name);
  const [error, setError] = useState<string | null>(null);

  function handleChange(
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    setDraft(event.target.value);
    setError(null);
  }

  function handleKeyDown(
    event: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    if (isKey(event, "Escape")) {
      event.preventDefault();
      onCancel();
      return;
    }
    if (!isKey(event, "Enter")) return;
    event.preventDefault();
    const next = draft.trim();
    if (!next) {
      setError("Give the column a name.");
      return;
    }
    onSave(next.slice(0, MAX_COLUMN_NAME));
  }

  return { draft, error, handleChange, handleKeyDown };
}
