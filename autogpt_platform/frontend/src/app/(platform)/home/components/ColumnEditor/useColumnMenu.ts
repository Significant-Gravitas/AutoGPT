import { useState } from "react";

type Action = "rename" | "add" | "delete";

interface Args {
  onRename: () => void;
  onAddAfter: () => void;
  onDelete: () => void;
}

export function useColumnMenu({ onRename, onAddAfter, onDelete }: Args) {
  const [chosen, setChosen] = useState<Action | null>(null);

  // Runs once the menu has closed and let go of focus, so the rename input
  // that the action opens keeps it. Delete lets focus return to this button
  // first, so the confirm dialog hands it back here when it closes.
  function handleCloseAutoFocus(event: Event) {
    if (!chosen) return;
    setChosen(null);
    if (chosen === "delete") {
      onDelete();
      return;
    }
    event.preventDefault();
    if (chosen === "rename") onRename();
    if (chosen === "add") onAddAfter();
  }

  return {
    handleCloseAutoFocus,
    chooseRename: () => setChosen("rename"),
    chooseAdd: () => setChosen("add"),
    chooseDelete: () => setChosen("delete"),
  };
}
