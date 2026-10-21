import type { KeyboardEvent as ReactKeyboardEvent } from "react";

type AnyKeyboardEvent = ReactKeyboardEvent | KeyboardEvent;

// The key names `isKey` accepts and the `no-restricted-syntax` selectors in
// .eslintrc.json guard — every key an IME (Japanese, Chinese, Korean, ...) can
// claim while composing, whether to confirm a candidate (Enter, Space), page
// through the candidate window (arrows, PageUp/PageDown, Home/End), cancel it
// (Escape), or edit the reading (Backspace, Delete, Tab).
//
// Keys outside this set are deliberately unguarded: an IME never claims them,
// so comparing `e.key` directly against e.g. "F2" is fine and the lint rule
// ignores it. The eslint-keyboard-rules.test.ts suite fails if this list and
// the selectors drift apart.
export const KEY_NAMES = [
  "Enter",
  "Escape",
  "Tab",
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  "PageUp",
  "PageDown",
  "Home",
  "End",
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
// itself, which arrives as key "Unidentified"/"Process" — never a KeyName, so
// `isKey` rejects it on the key comparison regardless. Their Enter arrives as
// keyCode 13 and is unaffected. cmdk applies the same predicate.
const IME_KEYCODE = 229;

/**
 * True while an IME is composing, so the keydown belongs to the input method
 * rather than to the app. Works on both React synthetic and native events.
 */
export function isComposingEvent(e: AnyKeyboardEvent): boolean {
  const native = "nativeEvent" in e ? e.nativeEvent : e;
  return native.isComposing || native.keyCode === IME_KEYCODE;
}

/**
 * True when the event matches one of `keys` and no IME composition is in
 * progress. Use this for every keyboard shortcut so a Japanese, Chinese or
 * Korean user confirming a candidate does not also trigger the app's action.
 */
export function isKey(e: AnyKeyboardEvent, ...keys: KeyName[]): boolean {
  return !isComposingEvent(e) && (keys as string[]).includes(e.key);
}

/**
 * True when the event matches one of `keys`, whether or not an IME is
 * composing. Only for handlers that enforce containment rather than trigger an
 * action — a focus trap must keep holding Tab during composition, since letting
 * it through moves focus out of the modal entirely. Prefer {@link isKey}.
 */
export function isKeyIgnoringComposition(
  e: AnyKeyboardEvent,
  ...keys: KeyName[]
): boolean {
  return (keys as string[]).includes(e.key);
}
