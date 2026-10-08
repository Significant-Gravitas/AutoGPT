import { isComposingEvent } from "@/lib/keyboard";

interface Dismissal {
  reason: string;
  event: Event;
}

/** An external picker (Google Drive) is open above the dialog. */
export function isExternalPickerOpen() {
  return document.body.hasAttribute("data-google-picker-open");
}

/** Escape pressed while an IME composes dismisses the candidate window, not the dialog. */
export function isComposingEscape({ reason, event }: Dismissal) {
  return (
    reason === "escape-key" &&
    event instanceof KeyboardEvent &&
    isComposingEvent(event)
  );
}

/** Outside presses and focus loss belong to the picker while it is open. */
export function isPickerInteraction({ reason }: Dismissal) {
  return (
    (reason === "outside-press" || reason === "focus-out") &&
    isExternalPickerOpen()
  );
}
