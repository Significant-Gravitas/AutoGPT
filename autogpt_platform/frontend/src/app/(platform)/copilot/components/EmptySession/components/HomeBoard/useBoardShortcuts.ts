import { isComposingEvent } from "@/lib/keyboard";
import { useEffect, type RefObject } from "react";

interface Args {
  onNew: () => void;
  searchRef: RefObject<HTMLInputElement | null>;
}

const NEW_TASK = "n";
const SEARCH = "/";

/** Whether a key belongs to what has focus: a field, a menu, a dialog. */
function isTaken(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return Boolean(
    target.closest(
      "input, textarea, select, [role='dialog'], [role='menu'], [role='listbox']",
    ),
  );
}

/** On the Tasks view, N opens the create dialog and / goes to search,
 *  unless focus is in a field, a menu or a dialog, a modifier is held or
 *  an input method is composing. */
export function useBoardShortcuts({ onNew, searchRef }: Args) {
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.defaultPrevented || isComposingEvent(e)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (isTaken(e.target)) return;
      const key = e.key.toLowerCase();
      if (key === NEW_TASK && !e.shiftKey) {
        e.preventDefault();
        onNew();
      } else if (key === SEARCH) {
        e.preventDefault();
        searchRef.current?.focus();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onNew, searchRef]);
}

/** The element the skip link and the shortcuts lead to. */
export const TASKS_REGION_ID = "home-tasks";
