import { useEffect } from "react";
import { useCopyPasteStore } from "../stores/copyPasteStore";

export function useCopyPasteKeyboard() {
  const { copySelectedNodes, pasteNodes } = useCopyPasteStore();

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const activeElement = document.activeElement;
      const isInputField =
        activeElement?.tagName === "INPUT" ||
        activeElement?.tagName === "TEXTAREA" ||
        activeElement?.getAttribute("contenteditable") === "true";

      if (isInputField) return;

      if (
        (event.ctrlKey || event.metaKey) &&
        (event.key === "c" || event.key === "C")
      ) {
        event.preventDefault();
        copySelectedNodes();
      }

      if (
        (event.ctrlKey || event.metaKey) &&
        (event.key === "v" || event.key === "V")
      ) {
        event.preventDefault();
        pasteNodes();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [copySelectedNodes, pasteNodes]);
}
