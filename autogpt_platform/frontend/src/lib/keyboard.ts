import type { KeyboardEvent as ReactKeyboardEvent } from "react";

type AnyKeyboardEvent = ReactKeyboardEvent | KeyboardEvent;

// While an IME (Japanese, Chinese, Korean, ...) is composing, Enter/Space/arrows/
// Escape belong to the input method, not to the app. Browsers flag those keydowns
// with `isComposing`; Safari can also deliver the confirming Enter after
// compositionend with `keyCode` 229 and `isComposing` already false.
export function isComposingEvent(e: AnyKeyboardEvent): boolean {
  const native = "nativeEvent" in e ? e.nativeEvent : e;
  return native.isComposing || native.keyCode === 229;
}

export function isKey(e: AnyKeyboardEvent, key: string): boolean {
  return !isComposingEvent(e) && e.key === key;
}
