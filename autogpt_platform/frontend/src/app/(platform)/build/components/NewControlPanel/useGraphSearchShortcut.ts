import { useControlPanelStore } from "@/app/(platform)/build/stores/controlPanelStore";
import { useEffect } from "react";

export function useGraphSearchShortcut() {
  const { setGraphSearchOpen } = useControlPanelStore();

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const el = document.activeElement as HTMLElement | null;
      if (
        el &&
        (el.tagName === "INPUT" ||
          el.tagName === "TEXTAREA" ||
          el.isContentEditable)
      ) {
        return;
      }

      if ((e.metaKey || e.ctrlKey) && e.key === "f") {
        e.preventDefault();
        setGraphSearchOpen(true);
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [setGraphSearchOpen]);
}
