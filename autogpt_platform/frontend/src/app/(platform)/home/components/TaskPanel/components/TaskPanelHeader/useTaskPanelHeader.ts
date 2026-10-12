import { isKey } from "@/lib/keyboard";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useState } from "react";

interface Args {
  title: string;
  isStarted: boolean;
  onRename: (title: string) => void;
}

export function useTaskPanelHeader({ title, isStarted, onRename }: Args) {
  const [draft, setDraft] = useState<string | null>(null);
  // Without experts the only assignee is Otto, so there is no one to hand to.
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);

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
    canReassign: isStarted && isExpertsEnabled,
    draft: draft ?? "",
    startEditing,
    stopEditing,
    handleChange,
    handleKeyDown,
  };
}
