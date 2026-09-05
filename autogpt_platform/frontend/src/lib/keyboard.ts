import type { KeyboardEvent as ReactKeyboardEvent } from "react";

type AnyKeyboardEvent = ReactKeyboardEvent | KeyboardEvent;

// Keys an IME (Japanese, Chinese, Korean, ...) claims while composing. Kept in
// sync with the `no-restricted-syntax` selectors in .eslintrc.json; the
// keyboard.test.ts suite fails if the two lists drift.
export const KEY_NAMES = [
  "Enter",
  "Escape",
  "Tab",
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  "Backspace",
  "Delete",
  " ",
] as const;

export type KeyName = (typeof KEY_NAMES)[number];

// Legacy keyCode browsers report for a keydown the IME handled. Safari also
// uses it for the Enter that confirms a candidate, which it fires after
// compositionend with `isComposing` already false, so `isComposing` alone
// would let that Enter through as a submit.
//
// Android soft keyboards report 229 too, but only for the composed text
// itself (key "Unidentified"/"Process", `isComposing` true); their Enter
// arrives as keyCode 13 and is unaffected. cmdk applies the same predicate.
const IME_KEYCODE = 229;

export function isComposingEvent(e: AnyKeyboardEvent): boolean {
  const native = "nativeEvent" in e ? e.nativeEvent : e;
  return native.isComposing || native.keyCode === IME_KEYCODE;
}

export function isKey(e: AnyKeyboardEvent, ...keys: KeyName[]): boolean {
  return !isComposingEvent(e) && (keys as string[]).includes(e.key);
}
